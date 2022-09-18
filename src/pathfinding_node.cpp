#include <cmath>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>

#include <ros/ros.h>
#include <std_msgs/Int16.h>
#include <tf2_ros/transform_listener.h>

#include <cstdint>
#include <mutex>
#include <opencv2/opencv.hpp>

#include "octomap/AbstractOcTree.h"
#include "octomap/OcTree.h"
#include "octomap/OcTreeKey.h"
#include "opencv2/core/hal/interface.h"
#include "opencv2/imgproc.hpp"
#include "ros/duration.h"
#include "ros/node_handle.h"
#include "std_msgs/String.h"
#include "tf2/exceptions.h"
#include "tf2_ros/buffer.h"
#include "visualization_msgs/MarkerArray.h"
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Vector3.h>

tf2_ros::Buffer tfBuffer;

// ### Constants ###
// (TODO: Convert to configurable parameters)

// Map points within drone_z \pm mapSlicezthickness are used in 2D conversion
const double kMapSliceZThickness = 0.2;

// Min / max height of the map to be considered, to remove floor / ceiling
const double kMinHeight = 0.0, kMaxHeight = 3.0;

// Target coordinates in map frame
const cv::Point2d targetCoordMetric(5.0, 5.0);

// Drone radius and safety margin (in meters). Occupied cells will be expanded
// by kDroneRadius * 2 + kSafetymargin to avoid collisions
const double kDroneRadius = 0.2, kSafetyMargin = 0.2;

// ##### Publishers #####
image_transport::Publisher vis_pub;

// Extract the frontier points (in map image coordinate) given occupied and free
// cells
std::vector<cv::Point> findFrontierPoints(const cv::Mat &traversible,
                                          const cv::Mat &occupiedSafe,
                                          const cv::Point &droneCoordinates) {

  // Find contours of traversible_ region
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Point> relevantContour;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(traversible, contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_NONE);

  // Find which contour the drone is in
  for (const auto &contour : contours) {
    if (cv::pointPolygonTest(contour, droneCoordinates, false) > 0) {
      relevantContour = contour;
      break;
    }
  }

  if (relevantContour.size() == 0) {
    ROS_WARN("No contour points found");
    return std::vector<cv::Point>();
  }

  // Frontier points are contours that are not adjacent to an occupied
  // location Dilate with 3x3 kernel to extend occupied by 1 pixel
  cv::Mat kernel =
      cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(3, 3));
  cv::Mat occupiedContour;
  cv::dilate(occupiedSafe, occupiedContour, kernel);

  std::vector<cv::Point> frontier;

  for (const auto &pt : relevantContour) {
    if (occupiedContour.at<uchar>(pt) != 255)
      frontier.push_back(pt);
  }

  return frontier;
}

void octomapCallback(const octomap_msgs::Octomap &msg) {

  // Convert from message to OcTree
  const double kResolution = msg.resolution;
  octomap::ColorOcTree *mapPtr = new octomap::ColorOcTree(kResolution);
  octomap::AbstractOcTree *msgTree = octomap_msgs::binaryMsgToMap(msg);
  mapPtr = dynamic_cast<octomap::ColorOcTree *>(msgTree);
  ROS_INFO("Tree conversion done");

  // Extract (metric) bounds of the known space (occupied or free)
  // Useful for restricting z
  double xMax, yMax, zMax, xMin, yMin, zMin;
  mapPtr->getMetricMax(xMax, yMax, zMax);
  mapPtr->getMetricMin(xMin, yMin, zMin);
  ROS_INFO("Bounds extracted");
  ROS_INFO("Max: %.4f, %.4f, %.4f | Min: %.4f %.4f %.4f", xMax, yMax, zMax,
           xMin, yMin, zMin);

  // Maximum width / height of 2D map depends on the target point as well
  const double mapXMin = std::min(xMin, targetCoordMetric.x) - kResolution,
               mapYMin = std::min(yMin, targetCoordMetric.y) - kResolution,
               mapXMax = std::max(xMax, targetCoordMetric.x) + kResolution,
               mapYMax = std::max(yMax, targetCoordMetric.y) + kResolution;

  // Obtain UAV location wrt to the map frame. Exit if cannot be retrieved
  geometry_msgs::Vector3 map2UAVPos;
  try {
    geometry_msgs::TransformStamped pose =
        tfBuffer.lookupTransform("map", "m100/base_link", ros::Time(0));
    map2UAVPos = pose.transform.translation;

  } catch (tf2::TransformException &ex) {
    ROS_WARN("%s", ex.what());
    ROS_INFO("Pose could not be computed. Exiting");
    return;
  }

  // Coordinates in the image frame
  size_t xCoordDrone = static_cast<size_t>(
             std::round((map2UAVPos.x - mapXMin) / kResolution)),
         yCoordDrone = static_cast<size_t>(
             std::round((map2UAVPos.y - mapYMin) / kResolution));
  cv::Point coordDrone(xCoordDrone, yCoordDrone);

  ROS_INFO("Pose computed: x: %.4f, y: %.4f, z: %.4f", map2UAVPos.x,
           map2UAVPos.y, map2UAVPos.z);

  // Safety radius in image frame
  const int droneSafetyRadius =
      std::ceil((kDroneRadius * 2.0 + kSafetyMargin) / kResolution);
  cv::Mat kernelSafety =
      cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE,
                                cv::Size(droneSafetyRadius, droneSafetyRadius));

  // Set bounds of the map to be extracted
  octomap::point3d maxBoundsZRestricted(
      xMax + 1.0, yMax + 1.0,
      std::min(map2UAVPos.z + kMapSliceZThickness, kMaxHeight)),
      minBoundsZRestricted(
          xMin - 1.0, yMin - 1.0,
          std::max(map2UAVPos.z - kMapSliceZThickness, kMinHeight));

  // Project the 3D bbx region into a 2D map (occupied / free)
  const size_t width = static_cast<size_t>(
                   std::ceil((mapXMax - mapXMin) / kResolution)),
               height = static_cast<size_t>(
                   std::ceil((mapYMax - mapYMin) / kResolution));

  cv::Mat occupied(height, width, CV_8UC1, cv::Scalar(0)),
      free(height, width, CV_8UC1, cv::Scalar(0)),
      zero(height, width, CV_8UC1, cv::Scalar(0));

  // Drone coordinate (and surrounding radius) is occupied
  free.at<uint8_t>(yCoordDrone, xCoordDrone) = 255;
  cv::dilate(free, free, kernelSafety);

  // iterate over bounding-box-restricted points & update occupied / free
  size_t count = 0;
  for (octomap::ColorOcTree::leaf_bbx_iterator
           it = mapPtr->begin_leafs_bbx(minBoundsZRestricted,
                                        maxBoundsZRestricted),
           end = mapPtr->end_leafs_bbx();
       it != end; ++it) {
    ++count;
    size_t xCoord = static_cast<size_t>(
               std::round((it.getX() - mapXMin) / kResolution)),
           yCoord = static_cast<size_t>(
               std::round((it.getY() - mapYMin) / kResolution));

    // If logOdd > 0 -> Occupied. Otherwise free
    if (it->getLogOdds() > 0) {
      occupied.at<uint8_t>(yCoord, xCoord) = 255;
    } else {
      free.at<uint8_t>(yCoord, xCoord) = 255;
    }
  }
  ROS_INFO("Total pts in bounding volume: %d", static_cast<int>(count));

  // Perform morphological closing on free map to eliminate small holes
  cv::Mat kernel3x3 =
      cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3));
  cv::morphologyEx(free, free, cv::MORPH_CLOSE, kernel3x3);

  // Dilate occupied cells to provide buffer for collision
  cv::Mat occupiedSafe, occupiedSafeMask;
  cv::dilate(occupied, occupiedSafe, kernelSafety);
  cv::threshold(occupiedSafe, occupiedSafeMask, 100, 1, cv::THRESH_BINARY_INV);

  // Traversible: min. distance to an obstacle is 2 drone radii away
  // Traversible safe: min. dist is ~3 drone radii
  cv::Mat traversible = free.mul(occupiedSafeMask);

  // Extract frontier points
  std::vector<cv::Point> frontier =
      findFrontierPoints(traversible, occupiedSafe, coordDrone);

  // ##### Visualization #####

  // Base image
  // Traversible region -> Blue (appears as cyan)
  // Free region -> Green (yellow / cyan as well)
  // Occupied region -> Red (some yellow)
  cv::Mat visual;
  std::vector<cv::Mat> channels{traversible, free, occupied};
  cv::merge(channels, visual);

  // Add drone location on map (blue circle)
  size_t radius = static_cast<size_t>(kDroneRadius / kResolution);
  cv::circle(visual, cv::Point(xCoordDrone, yCoordDrone), radius,
             cv::Scalar(255, 0, 0), 1);

  // Add target location on map (purple)
  size_t xCoordTarget = static_cast<size_t>(
             std::round((targetCoordMetric.x - mapXMin) / kResolution)),
         yCoordTarget = static_cast<size_t>(
             std::round((targetCoordMetric.y - mapYMin) / kResolution));
  cv::circle(visual, cv::Point(xCoordTarget, yCoordTarget), 0,
             cv::Scalar(255, 0, 255), -1);

  // Mark frontier points - Gray
  for (const auto &pt : frontier)
    cv::circle(visual, pt, 0, cv::Scalar(100, 100, 100), 1);

  // Correct orientation
  cv::flip(visual, visual, 0);

  sensor_msgs::ImagePtr visMsg =
      cv_bridge::CvImage(std_msgs::Header(), "bgr8", visual).toImageMsg();
  vis_pub.publish(visMsg);

  // Free map pointers
  delete mapPtr;
  return;
}

int main(int argc, char **argv) {
  const std::string nodeName = "pathfinding_node";
  ros::init(argc, argv, nodeName);
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  tf2_ros::TransformListener tfListener(tfBuffer);
  image_transport::ImageTransport imageTransport(nh);
  vis_pub = imageTransport.advertise("debug_vis", 1);

  ros::Subscriber octomapMsgSubscriber =
      nh.subscribe("/rtabmap/octomap_binary", 1, octomapCallback);

  ros::spin();

  return 0;
}
