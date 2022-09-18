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

// Drone radius (in meters, for safety margins & visualizations)
const double kDroneRadius = 0.2;

// ##### Publishers #####
image_transport::Publisher vis_pub;

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
  ROS_INFO("Pose computed: x: %.4f, y: %.4f, z: %.4f", map2UAVPos.x,
           map2UAVPos.y, map2UAVPos.z);

  // Set bounds of the map to be extracted
  octomap::point3d maxBoundsZRestricted(
      xMax + 1.0, yMax + 1.0,
      std::min(map2UAVPos.z + kMapSliceZThickness, kMaxHeight)),
      minBoundsZRestricted(
          xMin - 1.0, yMin - 1.0,
          std::max(map2UAVPos.z - kMapSliceZThickness, kMinHeight));

  // Generate a colored repr of the map in 2D
  // Maximum width / height depends on the target point as well
  const double mapXMin = std::min(xMin, targetCoordMetric.x) - kResolution,
               mapYMin = std::min(yMin, targetCoordMetric.y) - kResolution,
               mapXMax = std::max(xMax, targetCoordMetric.x) + kResolution,
               mapYMax = std::max(yMax, targetCoordMetric.y) + kResolution;

  const size_t width = static_cast<size_t>(
                   std::ceil((mapXMax - mapXMin) / kResolution)),
               height = static_cast<size_t>(
                   std::ceil((mapYMax - mapYMin) / kResolution));

  cv::Mat occupied(height, width, CV_8UC1, cv::Scalar(0)),
      free(height, width, CV_8UC1, cv::Scalar(0)),
      zero(height, width, CV_8UC1, cv::Scalar(0));
  size_t count = 0;

  // Iterate over bounding-box-restricted points & update occupied / free
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
      // Unmark free cells at this coordinate
      free.at<uint8_t>(yCoord, xCoord) = 0;
    } else {
      if (occupied.at<uint8_t>(yCoord, xCoord) != 255)
        free.at<uint8_t>(yCoord, xCoord) = 255;
    }
  }

  ROS_INFO("Total pts in range: %d", static_cast<int>(count));

  // ##### Visualization #####
  cv::Mat visual;
  std::vector<cv::Mat> channels{zero, free, occupied};
  cv::merge(channels, visual);

  // Add drone location on map (blue circle)
  size_t xCoordDrone = static_cast<size_t>(
             std::round((map2UAVPos.x - mapXMin) / kResolution)),
         yCoordDrone = static_cast<size_t>(
             std::round((map2UAVPos.y - mapYMin) / kResolution));
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
