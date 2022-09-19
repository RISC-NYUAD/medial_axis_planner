#include <cmath>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <limits>
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
#include "opencv2/core/cvdef.h"
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
const cv::Point2d posTarget(5.0, -5.0);

// Drone radius and safety margin (in meters). Occupied cells will be expanded
// by kDroneRadius * 2 + kSafetymargin to avoid collisions
const double kDroneRadius = 0.2, kSafetyMargin = 0.2;

// Neighborhood order in x and y: From top, clockwise
const int neighborX[8] = {0, 1, 1, 1, 0, -1, -1, -1},
          neighborY[8] = {-1, -1, 0, 1, 1, 1, 0, -1};

// ##### Publishers #####
image_transport::Publisher vis_pub;

// Extract the contour of traversible region that the drone is in
std::vector<cv::Point>
extractTraversibleContour(const cv::Mat &traversible,
                          const cv::Mat &occupiedSafe,
                          const cv::Point &droneCoordinates, bool simplify) {

  // Find contours of traversible_ region
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Point> relevantContour;
  std::vector<cv::Vec4i> hierarchy;

  if (simplify)
    cv::findContours(traversible, contours, hierarchy, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);
  else
    cv::findContours(traversible, contours, hierarchy, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_NONE);

  // Select the contour the drone is in
  for (const auto &contour : contours) {
    if (cv::pointPolygonTest(contour, droneCoordinates, false) > 0) {
      relevantContour = contour;
      break;
    }
  }

  return relevantContour;
}

// Extract the frontier points (in map image coordinate) given occupied and free
// cells
std::vector<cv::Point> findFrontierPoints(const cv::Mat &traversible,
                                          const cv::Mat &occupiedSafe,
                                          const cv::Point &droneCoordinates) {

  std::vector<cv::Point> contour = extractTraversibleContour(
      traversible, occupiedSafe, droneCoordinates, false);
  if (contour.size() == 0) {
    ROS_WARN("No contour points found");
    return std::vector<cv::Point>();
  }

  // Frontier points are contours of traversible space that are not adjacent to
  // an occupied location.
  // Dilate with 3x3 kernel to extend occupied by 1 pixel
  cv::Mat kernel =
      cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(3, 3));
  cv::Mat occupiedContour;
  cv::dilate(occupiedSafe, occupiedContour, kernel);

  std::vector<cv::Point> frontier;

  for (const auto &pt : contour) {
    if (occupiedContour.at<uchar>(pt) != 255)
      frontier.push_back(pt);
  }

  return frontier;
}

// Compute the cost map from target point to all frontier points using
// breadth-first search
cv::Mat computeCostMap(const std::vector<cv::Point> &frontier,
                       const cv::Mat &traversible, const cv::Mat &occupiedSafe,
                       const std::vector<cv::Point> &targetCoordinates,
                       const cv::Point &droneCoordinates) {

  // Extract simplified contour of occupied safe region
  // Find contours of traversible_ region
  std::vector<std::vector<cv::Point>> contoursOccupiedSafe;
  std::vector<cv::Point> relevantContour;
  std::vector<cv::Vec4i> hierarchy;

  cv::findContours(occupiedSafe, contoursOccupiedSafe, hierarchy,
                   cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  // Combine contours from occupied safe
  std::vector<cv::Point> contourOccupiedSafe;
  for (std::vector<cv::Point> &contour : contoursOccupiedSafe)
    contourOccupiedSafe.insert(contourOccupiedSafe.end(), contour.begin(),
                               contour.end());

  if (contourOccupiedSafe.size() == 0) {
    ROS_ERROR("Cost map contour issues");
    return cv::Mat();
  }

  // Combine all points for convex hull calculation
  std::vector<cv::Point> hullPointSet;
  hullPointSet.insert(hullPointSet.end(), frontier.begin(), frontier.end());
  hullPointSet.insert(hullPointSet.end(), contourOccupiedSafe.begin(),
                      contourOccupiedSafe.end());
  hullPointSet.insert(hullPointSet.end(), targetCoordinates.begin(),
                      targetCoordinates.end());

  // ROS_INFO("Num. pts for hull calculation: %d",
  //          static_cast<int>(hullPointSet.size()));

  std::vector<cv::Point> hull;
  cv::convexHull(hullPointSet, hull);

  // Copy traversible region mask, add convex hull
  cv::Mat allowedRegion = traversible.clone();
  cv::fillConvexPoly(allowedRegion, hull, cv::Scalar(255));

  // Dilate to provide additional room for movement. This will form all the
  // points that the drone is allowed to fly in
  cv::Mat kernel =
      cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(5, 5));
  cv::dilate(allowedRegion, allowedRegion, kernel);

  // A rect to check if point is inside the image bounds
  cv::Rect imageBounds(cv::Point(), allowedRegion.size());

  // Distances matrix to populate. Start at max distance
  // 16-bits unsigned can deal with ~600m of path at 0.1m map resolution, which
  // is probably sufficient for our current specs
  cv::Mat distances(allowedRegion.rows, allowedRegion.cols, CV_16UC1,
                    cv::Scalar(std::numeric_limits<uint16_t>::max()));

  // Create a queue of cv coordinates for BFS search
  std::queue<cv::Point> pixelQueue;

  // Target coordinates have distance 0
  for (const cv::Point &coord : targetCoordinates) {
    pixelQueue.push(coord);
    distances.at<uint16_t>(coord.y, coord.x) = 0;
  }

  // Count number of frontier points reached during BFS, to terminate early
  size_t numFrontierPointsReached = 0;
  const size_t numFrontierPoints = frontier.size();

  while (!pixelQueue.empty()) {
    cv::Point currentPx = pixelQueue.front();
    // ROS_INFO("x: %d, y:%d", currentPx.pt.x, currentPx.pt.y);

    // Add neighboring points (in 8-neighborhood) if they are
    // -> within image (needed so that convex hull check can be performed
    // -> within convex hull
    // -> haven't been visited
    // -> not in occupiedSafe
    for (size_t i = 0; i < 8; ++i) {
      cv::Point neighborPt(currentPx.x + neighborX[i],
                           currentPx.y + neighborY[i]);

      bool ptIsInImage = imageBounds.contains(neighborPt);
      bool ptIsAllowed =
          allowedRegion.at<uint8_t>(neighborPt.y, neighborPt.x) == 255;
      bool ptIsNotVisited =
          distances.at<uint16_t>(neighborPt.y, neighborPt.x) ==
          std::numeric_limits<uint16_t>::max();
      bool ptIsNotInOccupiedSafe =
          occupiedSafe.at<uint8_t>(neighborPt.y, neighborPt.x) == 0;

      // ROS_INFO(
      //     "(%d, %d) ptIsInImage = %s, ptIsAllowed = %s, ptIsNotVisited = %s",
      //     neighborPt.x, neighborPt.y, ptIsInImage ? "True" : "False",
      //     ptIsAllowed ? "True" : "False", ptIsNotVisited ? "True" : "False");

      if (ptIsInImage && ptIsAllowed && ptIsNotVisited &&
          ptIsNotInOccupiedSafe) {

        uint16_t currentDist = distances.at<uint16_t>(currentPx.y, currentPx.x);
        uint16_t neighborDist = currentDist + 1;

        // Set distance early to mark visited
        distances.at<uint16_t>(neighborPt.y, neighborPt.x) = neighborDist;
        pixelQueue.push(neighborPt);

        // If it is a frontier node, increment the counter
        for (const auto &pt : frontier)
          if (currentPx == pt)
            ++numFrontierPointsReached;
      }
    }

    // Remove the currently processed node
    pixelQueue.pop();

    // Exit if all frontier points are reached
    if (numFrontierPointsReached == numFrontierPoints) {
      break;
    }
  }

  return distances;
}

std::vector<cv::Point> computeShortestPathFromCostmap(const cv::Mat &costMap,
                                                      const cv::Point &start) {
  // CANNOT FIND DISTANCE TO A PARTICULAR TARGET POINT. ONLY THE SHORTEST PATH
  // TO THE CLOSEST PATH POINT
  // DOES NOT INCLUDE start IN THE RETURNED PATH

  std::vector<cv::Point> path;

  // Ensure that start point has an assigned distance (reached)
  uint16_t currentDist = costMap.at<uint16_t>(start.y, start.x);

  if (currentDist == std::numeric_limits<uint16_t>::max()) {
    ROS_WARN(
        "Start (%d, %d) point is has not been reached. Cannot compute path",
        start.x, start.y);
    return path;
  }

  // Used in checking that the neighbor is in the image
  cv::Rect imageBounds(cv::Point(), costMap.size());

  // Initialize
  cv::Point currentPt = start, minDistPt = start;

  while (currentDist != 0) {

    // Iterate over all neighbors. One with shortest distance is added to path
    for (size_t i = 0; i < 8; ++i) {
      cv::Point neighbor(currentPt.x + neighborX[i],
                         currentPt.y + neighborY[i]);
      if (!imageBounds.contains(neighbor))
        continue;

      uint16_t neighborDist = costMap.at<uint16_t>(neighbor.y, neighbor.x);

      if (neighborDist < currentDist) {
        currentDist = neighborDist;
        minDistPt = neighbor;
      }
    }

    // Add minimum distance point to the path
    path.push_back(minDistPt);

    // DEBUG: currentPt should not be equal to minDistPt. Distances are
    // monotonic
    if (currentPt == minDistPt) {
      ROS_ERROR("Path computation error. Investigate");
      return std::vector<cv::Point>();
    }

    // Update currentPt with minDistPt
    currentPt = minDistPt;
  }

  return path;
}

void octomapCallback(const octomap_msgs::Octomap &msg) {

  // Convert from message to OcTree
  const double kResolution = msg.resolution;
  octomap::ColorOcTree *mapPtr = new octomap::ColorOcTree(kResolution);
  octomap::AbstractOcTree *msgTree = octomap_msgs::binaryMsgToMap(msg);
  mapPtr = dynamic_cast<octomap::ColorOcTree *>(msgTree);

  // Extract (metric) bounds of the known space (occupied or free)
  double xMax, yMax, zMax, xMin, yMin, zMin;
  mapPtr->getMetricMax(xMax, yMax, zMax);
  mapPtr->getMetricMin(xMin, yMin, zMin);

  // Maximum width / height of 2D map depends on the target point as well
  // Expanding to account for area surrounding occupied borders
  const double mapXMin = std::min(xMin, posTarget.x) - kResolution * 5,
               mapYMin = std::min(yMin, posTarget.y) - kResolution * 5,
               mapXMax = std::max(xMax, posTarget.x) + kResolution * 5,
               mapYMax = std::max(yMax, posTarget.y) + kResolution * 5;

  // Obtain UAV location wrt to the map frame. Exit if cannot be retrieved
  geometry_msgs::Vector3 map2UAVPos;
  try {
    geometry_msgs::TransformStamped pose =
        tfBuffer.lookupTransform("map", "m100/base_link", ros::Time(0));
    map2UAVPos = pose.transform.translation;

  } catch (tf2::TransformException &ex) {
    ROS_WARN("%s", ex.what());
    ROS_WARN("Pose could not be computed. Exiting");
    return;
  }

  // Coordinates in the image frame
  size_t xCoordDrone = static_cast<size_t>(
             std::round((map2UAVPos.x - mapXMin) / kResolution)),
         yCoordDrone = static_cast<size_t>(
             std::round((map2UAVPos.y - mapYMin) / kResolution));
  const cv::Point coordDrone(xCoordDrone, yCoordDrone);

  size_t xCoordTarget = static_cast<size_t>(
             std::round((posTarget.x - mapXMin) / kResolution)),
         yCoordTarget = static_cast<size_t>(
             std::round((posTarget.y - mapYMin) / kResolution));

  const cv::Point coordTarget(xCoordTarget, yCoordTarget);

  // Safety radius in image frame
  // Drone diameter + 1 extra radius + safety margin
  const int droneSafetyDiameter =
      std::ceil((kDroneRadius * 3.0 + kSafetyMargin) / kResolution);
  cv::Mat kernelSafety = cv::getStructuringElement(
      cv::MorphShapes::MORPH_ELLIPSE,
      cv::Size(droneSafetyDiameter, droneSafetyDiameter));

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
  for (octomap::ColorOcTree::leaf_bbx_iterator
           it = mapPtr->begin_leafs_bbx(minBoundsZRestricted,
                                        maxBoundsZRestricted),
           end = mapPtr->end_leafs_bbx();
       it != end; ++it) {
    size_t xCoord = static_cast<size_t>(
               std::round((it.getX() - mapXMin) / kResolution)),
           yCoord = static_cast<size_t>(
               std::round((it.getY() - mapYMin) / kResolution));

    // If logOdd > 0 -> Occupied. Otherwise free
    // Checks for overlapping free / occupied is not essential
    if (it->getLogOdds() > 0) {
      occupied.at<uint8_t>(yCoord, xCoord) = 255;
      free.at<uint8_t>(yCoord, xCoord) = 0;
    } else {
      if (occupied.at<uint8_t>(yCoord, xCoord) == 0)
        free.at<uint8_t>(yCoord, xCoord) = 255;
    }
  }

  // Perform morphological closing on free map to eliminate small holes
  cv::Mat kernel3x3 =
      cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3));
  cv::morphologyEx(free, free, cv::MORPH_CLOSE, kernel3x3);

  // Dilate occupied cells to provide buffer for collision
  cv::Mat occupiedSafe, occupiedSafeMask;
  cv::dilate(occupied, occupiedSafe, kernelSafety);
  cv::threshold(occupiedSafe, occupiedSafeMask, 100, 1, cv::THRESH_BINARY_INV);

  // Traversible: min. distance to an obstacle is 2 drone radii away
  // Traversible safe: min. dist to obstacles is ~3 drone radii from drone
  // center
  cv::Mat traversible = free.mul(occupiedSafeMask);

  // Extract frontier points
  std::vector<cv::Point> frontier =
      findFrontierPoints(traversible, occupiedSafe, coordDrone);

  if (frontier.empty())
    return;

  // Calculate distance map
  std::vector<cv::Point> targetPoints{coordTarget};
  cv::Mat costMap = computeCostMap(frontier, traversible, occupiedSafe,
                                   targetPoints, coordDrone);

  // Return if cost map is empty
  if (costMap.empty()) {
    ROS_ERROR("Cost map empty. Investigate");
    return;
  }

  // Compute path from the frontier point with the shortest distance
  cv::Point selectedFrontierPt = frontier[0];
  for (const cv::Point &pt : frontier) {
    if (costMap.at<uint16_t>(pt.y, pt.x) <
        costMap.at<uint16_t>(selectedFrontierPt.y, selectedFrontierPt.x)) {
      selectedFrontierPt = pt;
    }
  }
  std::vector<cv::Point> shortestPathFromFrontier =
      computeShortestPathFromCostmap(costMap, selectedFrontierPt);

  // ##### Visualization #####

  // Cost map
  cv::Mat costMapVis = costMap.clone(), costMapInvalidMask;

  // Replace uint16_t max with 0
  cv::threshold(costMapVis, costMapInvalidMask,
                std::numeric_limits<uint16_t>::max() - 2, 1,
                cv::THRESH_BINARY_INV);
  costMapVis = costMapVis.mul(costMapInvalidMask);

  // Put images in range of 0 - 255 in 8 bits
  cv::normalize(costMapVis, costMapVis, 0, 255, cv::NORM_MINMAX);
  costMapVis.convertTo(costMapVis, CV_8UC1);

  // Remove overlapping costmap from traversible area
  cv::Mat traversibleInverseMask;
  cv::threshold(traversible, traversibleInverseMask, 100, 1,
                cv::THRESH_BINARY_INV);
  costMapVis = costMapVis.mul(traversibleInverseMask);

  // cv::cvtColor(costMapVis8bit, costMapVis8bit, cv::COLOR_GRAY2BGR);
  // sensor_msgs::ImagePtr costMapMsg =
  //     cv_bridge::CvImage(std_msgs::Header(), "bgr8", costMapVis8bit)
  //         .toImageMsg();
  // vis_pub.publish(costMapMsg);

  // Base image
  // Traversible region -> Green
  // Frontier cost map -> Green
  // Occupied region -> Red (exact borders)
  cv::Mat visual;
  // std::vector<cv::Mat> channels{traversible, free, occupied};
  std::vector<cv::Mat> channels{costMapVis, traversible, occupied};
  cv::merge(channels, visual);

  // Add drone location on map (magenta circle)
  size_t radius = static_cast<size_t>(kDroneRadius / kResolution);
  cv::circle(visual, cv::Point(xCoordDrone, yCoordDrone), radius,
             cv::Scalar(255, 0, 255), 1);

  // Add target location on map (purple)
  cv::circle(visual, cv::Point(xCoordTarget, yCoordTarget), 0,
             cv::Scalar(155, 0, 155), -1);

  // Mark frontier points - White
  for (const auto &pt : frontier)
    cv::circle(visual, pt, 0, cv::Scalar(255, 255, 255), 1);

  // Mark shortest path to target from frontier - Yellow
  for (const auto &pt : shortestPathFromFrontier)
    cv::circle(visual, pt, 0, cv::Scalar(0, 255, 255), 1);

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
