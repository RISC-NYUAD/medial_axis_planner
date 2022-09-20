#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/Vector3.h>
#include <image_transport/image_transport.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
#include <ros/ros.h>
#include <std_msgs/Int16.h>
#include <tf2_ros/transform_listener.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <queue>

#include "geometry_msgs/Quaternion.h"
#include "octomap/AbstractOcTree.h"
#include "octomap/OcTree.h"
#include "octomap/OcTreeKey.h"
#include "opencv2/core.hpp"
#include "opencv2/core/cvdef.h"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/types.hpp"
#include "opencv2/imgproc.hpp"
#include "ros/duration.h"
#include "ros/node_handle.h"
#include "ros/publisher.h"
#include "std_msgs/String.h"
#include "tf2/exceptions.h"
#include "tf2_ros/buffer.h"
#include "timer.hpp"
#include "visualization_msgs/MarkerArray.h"

tf2_ros::Buffer tfBuffer;

// ### Constants ###
// (TODO: Convert to configurable parameters)

// Map points within drone_z \pm mapSlicezthickness are used in 2D conversion
const double kMapSliceZThickness = 0.2;

// Min / max height of the map to be considered, to remove floor / ceiling
const double kMinHeight = 0.0, kMaxHeight = 3.0;

// Target coordinates in map frame
const cv::Point2d posTarget(6.0, -5.0);

// Drone radius and safety margin (in meters). Occupied cells will be expanded
// by kDroneRadius * 2 + kSafetymargin to avoid collisions
const double kDroneRadius = 0.2, kSafetyMargin = 0.1;

// Navigation height (z axis value)
const double kNavigationHeight = 1.0;

// Timing information debugging
const bool debug = true;

// ##### Publishers #####
image_transport::Publisher debugVis;  // Debug visualization
ros::Publisher pathPub;               // Publishes path from uav to frontier

// Compute 8-neighbors of a given point
std::vector<cv::Point> computeNeighbors(const cv::Point& pt) {
  // Neighborhood order in x and y: From top, clockwise
  const int neighborX[8] = {0, 1, 1, 1, 0, -1, -1, -1},
            neighborY[8] = {-1, -1, 0, 1, 1, 1, 0, -1};

  std::vector<cv::Point> neighborhood;

  for (size_t i = 0; i < 8; ++i) {
    cv::Point neighborPt(pt.x + neighborX[i], pt.y + neighborY[i]);
    neighborhood.push_back(neighborPt);
  }
  return neighborhood;
}

// Extract the contour of traversible region that the drone is in
std::vector<cv::Point> extractTraversibleContour(const cv::Mat& traversible,
                                                 const cv::Point& droneCoordinates,
                                                 bool simplify) {
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
  for (const auto& contour : contours) {
    if (cv::pointPolygonTest(contour, droneCoordinates, false) > 0) {
      relevantContour = contour;
      break;
    }
  }

  return relevantContour;
}

// Extract the frontier points (in map image coordinate) given occupied and free
// cells
std::vector<cv::Point> findFrontierPoints(const cv::Mat& traversible,
                                          const cv::Mat& occupiedSafe,
                                          const cv::Point& droneCoordinates) {
  std::vector<cv::Point> contour =
      extractTraversibleContour(traversible, droneCoordinates, false);
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

  for (const auto& pt : contour) {
    if (occupiedContour.at<uchar>(pt) != 255)
      frontier.push_back(pt);
  }

  return frontier;
}

// Compute the cost map from target point to all frontier points using
// breadth-first search
cv::Mat computeCostMap(const std::vector<cv::Point>& frontier,
                       const cv::Mat& traversible,
                       const cv::Mat& occupiedSafe,
                       const std::vector<cv::Point>& targetCoordinates,
                       const cv::Point& droneCoordinates) {
  // Extract simplified contour of occupied safe region
  // Find contours of traversible_ region
  std::vector<std::vector<cv::Point>> contoursOccupiedSafe;
  std::vector<cv::Point> relevantContour;
  std::vector<cv::Vec4i> hierarchy;

  cv::findContours(occupiedSafe, contoursOccupiedSafe, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  // Combine contours from occupied safe
  std::vector<cv::Point> contourOccupiedSafe;
  for (std::vector<cv::Point>& contour : contoursOccupiedSafe)
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
  for (const cv::Point& coord : targetCoordinates) {
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
    for (const cv::Point& neighborPt : computeNeighbors(currentPx)) {
      bool ptIsInImage = imageBounds.contains(neighborPt);
      bool ptIsAllowed = allowedRegion.at<uint8_t>(neighborPt.y, neighborPt.x) == 255;
      bool ptIsNotVisited = distances.at<uint16_t>(neighborPt.y, neighborPt.x) ==
                            std::numeric_limits<uint16_t>::max();
      bool ptIsNotInOccupiedSafe =
          occupiedSafe.at<uint8_t>(neighborPt.y, neighborPt.x) == 0;

      // ROS_INFO(
      //     "(%d, %d) ptIsInImage = %s, ptIsAllowed = %s, ptIsNotVisited = %s",
      //     neighborPt.x, neighborPt.y, ptIsInImage ? "True" : "False",
      //     ptIsAllowed ? "True" : "False", ptIsNotVisited ? "True" : "False");

      if (ptIsInImage && ptIsAllowed && ptIsNotVisited && ptIsNotInOccupiedSafe) {
        uint16_t currentDist = distances.at<uint16_t>(currentPx.y, currentPx.x);
        uint16_t neighborDist = currentDist + 1;

        // Set distance early to mark visited
        distances.at<uint16_t>(neighborPt.y, neighborPt.x) = neighborDist;
        pixelQueue.push(neighborPt);

        // If it is a frontier node, increment the counter
        for (const auto& pt : frontier)
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

std::vector<cv::Point> computeShortestPathFromCostmap(const cv::Mat& costMap,
                                                      const cv::Point& start) {
  // CANNOT FIND DISTANCE TO A PARTICULAR TARGET POINT. ONLY THE SHORTEST PATH
  // TO THE CLOSEST PATH POINT
  // DOES NOT INCLUDE start IN THE RETURNED PATH

  std::vector<cv::Point> path;

  // Ensure that start point has an assigned distance (reached)
  uint16_t currentDist = costMap.at<uint16_t>(start.y, start.x);

  if (currentDist == std::numeric_limits<uint16_t>::max()) {
    ROS_WARN("Start (%d, %d) point is has not been reached. Cannot compute path",
             start.x, start.y);
    return path;
  }

  // Used in checking that the neighbor is in the image
  cv::Rect imageBounds(cv::Point(), costMap.size());

  // Initialize
  cv::Point currentPt = start;

  while (currentDist != 0) {
    // Iterate over all neighbors. One with shortest distance is added to path
    for (const cv::Point& neighborPt : computeNeighbors(currentPt)) {
      // Cannot read distance if point is outside image
      if (!imageBounds.contains(neighborPt))
        continue;

      uint16_t neighborDist = costMap.at<uint16_t>(neighborPt.y, neighborPt.x);

      if (neighborDist < currentDist) {
        currentDist = neighborDist;
        currentPt = neighborPt;
      }
    }

    // Add minimum distance point to the path
    path.push_back(currentPt);
  }

  return path;
}

std::vector<cv::Point> computeLineOfSightPath(const cv::Mat& traversible,
                                              const cv::Point& start,
                                              const cv::Point& end) {
  // Checks if the linear trajectory between start and end points traverse
  // outside of the boundary defined by the image, and returns the straight
  // line if it is within the bounds
  std::vector<cv::Point> retval;
  cv::LineIterator it(traversible, start, end, 8);

  // Starting from 1 to exclude the starting point
  ++it;
  for (int i = 1; i < it.count; i++, ++it) {
    if (traversible.at<uint8_t>(it.pos()) != 255) {
      return std::vector<cv::Point>();
    } else {
      retval.push_back(it.pos());
    }
  }
  return retval;
}

std::vector<cv::Point> findPathAlongSkeleton(const std::vector<cv::Point>& skeletonPts,
                                             const cv::Point& start,
                                             const cv::Point& end) {
  // Struct to hold point and a distance
  struct PtWithDistance {
    size_t idx;
    cv::Point pt;
    // uint16_t dist; // For l-inf
    float dist;
  };

  // Comparator reversed for higher priority on small distances
  struct PtWithDistanceComparator {
    bool operator()(const PtWithDistance& lhs, const PtWithDistance& rhs) {
      return lhs.dist > rhs.dist;
    }
  };

  // Comparator for cv::Point types
  struct cvPointComparator {
    bool operator()(const cv::Point& lhs, const cv::Point& rhs) {
      if (lhs.x == rhs.x)
        return lhs.y < rhs.y;
      return lhs.x < rhs.x;
    }
  };

  // If end and start are the same, simply return one of them
  if (start == end)
    return std::vector<cv::Point>{start};

  // Keep track of distance and visit status to deal with updating requirements
  size_t N = skeletonPts.size();
  // std::vector<uint16_t> nodeDistances(N,
  // std::numeric_limits<uint16_t>::max());
  std::vector<float> nodeDistances(N, std::numeric_limits<float>::max());

  std::vector<bool> visited(N, false);

  // Previous path history to trace the path from one node to other
  std::vector<size_t> pathPrev(N, std::numeric_limits<size_t>::max());

  // Assign a mapping from point to their index
  std::map<cv::Point, size_t, cvPointComparator> ptToIdx;
  for (size_t i = 0; i < N; ++i) {
    ptToIdx[skeletonPts[i]] = i;
  }

  // Ensure that the start and end are actually inside the skeleton
  std::map<cv::Point, size_t>::iterator itStart, itEnd;
  itStart = ptToIdx.find(start);
  itEnd = ptToIdx.find(end);
  if (itStart == ptToIdx.end() || itEnd == ptToIdx.end()) {
    ROS_ERROR("Start or end point is not part of the skeleton. Investigate");
    return std::vector<cv::Point>();
  }

  // Distance-based priority queue
  std::priority_queue<PtWithDistance, std::vector<PtWithDistance>,
                      PtWithDistanceComparator>
      pq;

  // Start point has 0 distance
  nodeDistances[ptToIdx[start]] = 0;
  for (size_t i = 0; i < N; ++i)
    pq.push(PtWithDistance{i, skeletonPts[i], nodeDistances[i]});

  // Mark start to be visited
  visited[ptToIdx[start]] = true;

  while (!pq.empty()) {
    PtWithDistance curr = pq.top();
    pq.pop();

    // Ensure distance is up to date
    if (curr.dist != nodeDistances[curr.idx])
      continue;

    // Terminate if the end node is reached
    // NOTE: Does not work if the skeleton is comprised of 2 components. Need
    // all paths
    // if (curr.pt == end) break;

    visited[curr.idx] = true;

    std::vector<cv::Point> neighborhood = computeNeighbors(curr.pt);

    for (const cv::Point& neighbor : neighborhood) {
      // Skip if not in skeleton
      if (ptToIdx.find(neighbor) == ptToIdx.end())
        continue;

      size_t neighborIdx = ptToIdx[neighbor];

      // Skip if visited
      if (visited[neighborIdx])
        continue;

      // Using l-inf distance
      // uint16_t newDist = curr.dist + 1;

      // Using Euclidean distance
      cv::Point diff = curr.pt - neighbor;
      float dist = std::sqrt(static_cast<float>(diff.x * diff.x) +
                             static_cast<float>(diff.y * diff.y));
      float newDist = curr.dist + dist;

      if (newDist < nodeDistances[neighborIdx]) {
        nodeDistances[neighborIdx] = newDist;
        pq.push(PtWithDistance{neighborIdx, neighbor, newDist});
        pathPrev[neighborIdx] = curr.idx;
      }
    }
  }

  // Trace the path, if it exists'
  const size_t endIdx = ptToIdx[end], startIdx = ptToIdx[start];

  if (pathPrev[endIdx] == std::numeric_limits<size_t>::max()) {
    ROS_WARN("No path exists between given start and end points");
    return std::vector<cv::Point>();
  }

  // Calculate path from end to start
  std::vector<cv::Point> path;
  size_t curIdx = endIdx;
  while (curIdx != startIdx) {
    path.push_back(skeletonPts[curIdx]);
    curIdx = pathPrev[curIdx];
  }

  // Invert the path to orient from start to end
  std::reverse(path.begin(), path.end());
  return path;
}

void visualize(const cv::Mat& costMap,
               const cv::Mat& traversible,
               const cv::Mat& occupied,
               const std::vector<cv::Point>& skeletonPts,
               const cv::Point& coordDrone,
               const std::vector<cv::Point>& frontier,
               const std::vector<cv::Point>& uavToSkeleton,
               const std::vector<cv::Point>& alongSkeleton,
               const std::vector<cv::Point>& skeletonToTarget,
               const std::vector<cv::Point>& frontierToTarget,
               const cv::Point& coordTarget,
               const float& kResolution) {
  // ##### Visualization #####

  cv::Mat costMapVis;
  if (!costMap.empty()) {
    cv::Mat costMapInvalidMask;
    // Cost map
    costMapVis = costMap.clone();

    // Replace uint16_t max with 0
    cv::threshold(costMapVis, costMapInvalidMask,
                  std::numeric_limits<uint16_t>::max() - 2, 1, cv::THRESH_BINARY_INV);
    costMapVis = costMapVis.mul(costMapInvalidMask);

    // Put images in range of 0 - 255 in 8 bits
    cv::normalize(costMapVis, costMapVis, 0, 255, cv::NORM_MINMAX);
    costMapVis.convertTo(costMapVis, CV_8UC1);

    // Remove overlapping costmap from traversible area
    cv::Mat traversibleInverseMask;
    cv::threshold(traversible, traversibleInverseMask, 100, 1, cv::THRESH_BINARY_INV);
    costMapVis = costMapVis.mul(traversibleInverseMask);
  } else {
    costMapVis = cv::Mat(traversible.size(), CV_8UC1, cv::Scalar(0));
  }

  // Base image
  // Traversible region -> Green
  // Frontier cost map -> Blue
  // Occupied region -> Red (exact borders)
  // Black space is either occupied safe (around occupied regions), or not
  // discovered
  cv::Mat visual;
  // std::vector<cv::Mat> channels{traversible, free, occupied};
  std::vector<cv::Mat> channels{costMapVis, traversible, occupied};
  cv::merge(channels, visual);

  // Skeleton (gray)
  for (const auto& pt : skeletonPts) {
    cv::circle(visual, pt, 0, cv::Scalar(100, 100, 100), 1);
  }

  // Add drone location on map (magenta circle)
  size_t radius = static_cast<size_t>(kDroneRadius / kResolution);
  cv::circle(visual, coordDrone, radius, cv::Scalar(255, 0, 255), 1);

  // Mark frontier points (white)
  for (const auto& pt : frontier)
    cv::circle(visual, pt, 0, cv::Scalar(255, 255, 255), 1);

  // ### Paths ###

  // UAV -> Skeleton (blue-ish)
  for (const auto& pt : uavToSkeleton)
    cv::circle(visual, pt, 0, cv::Scalar(255, 155, 0), 1);

  // Along skeleton (light gray)
  for (const auto& pt : alongSkeleton)
    cv::circle(visual, pt, 0, cv::Scalar(200, 200, 200), 1);

  // Skeleton -> frontier / target (orange)
  for (const auto& pt : skeletonToTarget)
    cv::circle(visual, pt, 0, cv::Scalar(0, 155, 255), 1);

  // Frontier -> target (yellow)
  for (const auto& pt : frontierToTarget)
    cv::circle(visual, pt, 0, cv::Scalar(0, 255, 255), 1);

  // Add target location on map (purple)
  cv::circle(visual, coordTarget, 0, cv::Scalar(100, 0, 100), -1);

  // Correct orientation
  cv::flip(visual, visual, 0);

  sensor_msgs::ImagePtr visMsg =
      cv_bridge::CvImage(std_msgs::Header(), "bgr8", visual).toImageMsg();
  debugVis.publish(visMsg);
}

cv::Point findClosestLoSSkeletonPoint(const std::vector<cv::Point>& skeletonPts,
                                      const cv::Point& target,
                                      const cv::Mat& traversible) {
  // Find skeleton point closest to the selected frontier point with LoS visibility
  float currentDistance = std::numeric_limits<float>::max(),
        backupDistance = std::numeric_limits<float>::max();

  cv::Point closestPoint, backupPoint;
  bool setClosestPoint = false;

  for (const cv::Point& pt : skeletonPts) {
    cv::Point diff = pt - target;
    float dist = std::sqrt(diff.x * diff.x + diff.y * diff.y);

    // No LoS check / closest point
    if (dist < backupDistance) {
      backupPoint = pt;
      backupDistance = dist;
    }

    if (dist < currentDistance) {
      // LoS needed only if the distance is smaller
      std::vector<cv::Point> lineOfSightPath =
          computeLineOfSightPath(traversible, pt, target);

      // If LoS path is empty, then there is no path
      if (lineOfSightPath.empty()) {
        continue;
      } else {
        currentDistance = dist;
        setClosestPoint = true;
        closestPoint = pt;
      }
    }
  }

  if (setClosestPoint == false) {
    // Assigning a point here so that the returned point is still along the skeleton
    // NO LOS PATH EXISTS. THIS MAY BE DANGEROUS
    ROS_ERROR("No line-of-sight skeleton point exists. Returning closest point");
    return backupPoint;
  }

  return closestPoint;
}

void octomapCallback(const octomap_msgs::Octomap& msg) {
  Timer timer;
  timer.Tic();

  // Convert from message to OcTree
  const double kResolution = msg.resolution;
  octomap::ColorOcTree* mapPtr = new octomap::ColorOcTree(kResolution);
  octomap::AbstractOcTree* msgTree = octomap_msgs::binaryMsgToMap(msg);
  mapPtr = dynamic_cast<octomap::ColorOcTree*>(msgTree);

  ROS_INFO_COND(debug, "[TIMING] Message to OcTree: %.4f", timer.Toc());

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

  ROS_INFO_COND(debug, "[TIMING] Map bound extraction: %.4f", timer.Toc());

  // Obtain UAV location wrt to the map frame. Exit if cannot be retrieved
  geometry_msgs::TransformStamped uavPose;

  try {
    uavPose = tfBuffer.lookupTransform("map", "m100/base_link", ros::Time(0));

  } catch (tf2::TransformException& ex) {
    ROS_WARN("%s", ex.what());
    ROS_WARN("Pose could not be computed. Exiting");
    delete mapPtr;
    return;
  }

  ROS_INFO_COND(debug, "[TIMING] UAV pose extraction: %.4f", timer.Toc());

  // Coordinates in the image frame
  size_t xCoordDrone = static_cast<size_t>(
             std::round((uavPose.transform.translation.x - mapXMin) / kResolution)),
         yCoordDrone = static_cast<size_t>(
             std::round((uavPose.transform.translation.y - mapYMin) / kResolution));
  const cv::Point coordDrone(xCoordDrone, yCoordDrone);

  size_t xCoordTarget =
             static_cast<size_t>(std::round((posTarget.x - mapXMin) / kResolution)),
         yCoordTarget =
             static_cast<size_t>(std::round((posTarget.y - mapYMin) / kResolution));

  const cv::Point coordTarget(xCoordTarget, yCoordTarget);

  ROS_INFO_COND(debug, "[TIMING] UAV / Target coordinate to image computation: %.4f",
                timer.Toc());

  // Safety radius in image frame
  // Drone diameter + 0 extra radius + safety margin
  const int droneSafetyDiameter =
      std::ceil((kDroneRadius * 2.0 + kSafetyMargin) / kResolution);
  cv::Mat kernelSafety =
      cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE,
                                cv::Size(droneSafetyDiameter, droneSafetyDiameter));

  ROS_INFO_COND(debug, "[TIMING] Safety radius assignment: %.4f", timer.Toc());

  // ##### Project 3D-bounded map to a 2D binary occupancy
  // Set bounds of the map to be extracted
  octomap::point3d maxBoundsZRestricted(
      xMax + 1.0, yMax + 1.0,
      std::min(uavPose.transform.translation.z + kMapSliceZThickness, kMaxHeight)),
      minBoundsZRestricted(
          xMin - 1.0, yMin - 1.0,
          std::max(uavPose.transform.translation.z - kMapSliceZThickness, kMinHeight));

  // Project the 3D bbx region into a 2D map (occupied / free)
  const size_t width =
                   static_cast<size_t>(std::ceil((mapXMax - mapXMin) / kResolution)),
               height =
                   static_cast<size_t>(std::ceil((mapYMax - mapYMin) / kResolution));

  cv::Mat occupied(height, width, CV_8UC1, cv::Scalar(0)),
      free(height, width, CV_8UC1, cv::Scalar(0)),
      zero(height, width, CV_8UC1, cv::Scalar(0));

  // Drone coordinate (and surrounding radius) is occupied
  free.at<uint8_t>(yCoordDrone, xCoordDrone) = 255;
  cv::dilate(free, free, kernelSafety);

  // iterate over bounding-box-restricted points & update occupied / free
  for (octomap::ColorOcTree::leaf_bbx_iterator
           it = mapPtr->begin_leafs_bbx(minBoundsZRestricted, maxBoundsZRestricted),
           end = mapPtr->end_leafs_bbx();
       it != end; ++it) {
    size_t xCoord =
               static_cast<size_t>(std::round((it.getX() - mapXMin) / kResolution)),
           yCoord =
               static_cast<size_t>(std::round((it.getY() - mapYMin) / kResolution));

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

  ROS_INFO_COND(debug, "[TIMING] Projecting map to 2D: %.4f", timer.Toc());

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

  ROS_INFO_COND(debug, "[TIMING] 2D map processing: %.4f", timer.Toc());

  // ##### Path computation #####
  // Two cases: Target is inside traversible region that the UAV is in, or outside
  // if inside, simply traverse along the skeleton
  // if outside, find the closest frontier, and traverse there
  cv::Mat costMap;
  std::vector<cv::Point> path, skeletonPts, frontier, uavToSkeleton, alongSkeleton,
      skeletonToTarget, frontierToTarget;

  std::vector<cv::Point> traversibleRegionContainingUAV =
      extractTraversibleContour(free, coordDrone, true);

  if (traversibleRegionContainingUAV.empty()) {
    ROS_WARN("Empty contour. Cannot calculate path");
    visualize(costMap, traversible, occupied, skeletonPts, coordDrone, frontier,
              uavToSkeleton, alongSkeleton, skeletonToTarget, frontierToTarget,
              coordTarget, kResolution);
    delete mapPtr;
    return;
  }

  // Extract skeleton for the contour that contains the UAV
  cv::Mat skeleton,
      traversibleContourContainingUAVMask(traversible.size(), CV_8UC1, cv::Scalar(0));
  std::vector<std::vector<cv::Point>> contourForFilling{traversibleRegionContainingUAV};

  cv::fillPoly(traversibleContourContainingUAVMask, contourForFilling, cv::Scalar(1));

  cv::Mat traversibleContourContainingUAV =
      traversible.mul(traversibleContourContainingUAVMask);

  cv::ximgproc::thinning(traversibleContourContainingUAV, skeleton);
  ROS_INFO_COND(debug, "[TIMING] Medial axis on traversible region: %.4f", timer.Toc());

  // Mat to vector for skeleton points
  cv::findNonZero(skeleton, skeletonPts);

  ROS_INFO_COND(debug, "[TIMING] Skeleton image to skeleton coordinates: %.4f",
                timer.Toc());

  if (skeletonPts.empty()) {
    ROS_ERROR("Skeleton could not be computed. Investigate");
    visualize(costMap, traversible, occupied, skeletonPts, coordDrone, frontier,
              uavToSkeleton, alongSkeleton, skeletonToTarget, frontierToTarget,
              coordTarget, kResolution);
    delete mapPtr;
    return;
  }

  // Path from UAV to skeleton
  cv::Point entrySkeletonPt =
      findClosestLoSSkeletonPoint(skeletonPts, coordDrone, free);

  uavToSkeleton = computeLineOfSightPath(traversible, coordDrone, entrySkeletonPt);
  path.insert(path.end(), uavToSkeleton.begin(), uavToSkeleton.end());

  ROS_INFO_COND(debug,
                "[TIMING] Finding cloest LoS skeleton point to the UAV location: %.4f",
                timer.Toc());

  if (!traversibleRegionContainingUAV.empty() &&
      cv::pointPolygonTest(traversibleRegionContainingUAV, coordTarget, false) > 0) {
    ROS_INFO(">>> Target location is inside the traversible region");

    // Path from UAV to skeleton
    cv::Point exitSkeletonPt =
        findClosestLoSSkeletonPoint(skeletonPts, coordTarget, traversible);
    skeletonToTarget = computeLineOfSightPath(traversible, exitSkeletonPt, coordTarget);
    ROS_INFO_COND(
        debug,
        "[TIMING] Finding cloest LoS skeleton point to the target location: %.4f",
        timer.Toc());

    // Path along the skeleton between entryPoint and exitPoint
    alongSkeleton = findPathAlongSkeleton(skeletonPts, entrySkeletonPt, exitSkeletonPt);

    ROS_INFO_COND(debug, "[TIMING] Finding path along the skeleton: %.4f", timer.Toc());

  } else {
    ROS_INFO(">>> Target location is outside the traversible region");

    // ##### Find path from frontier to target #####
    // Extract frontier points
    frontier = findFrontierPoints(traversible, occupiedSafe, coordDrone);

    ROS_INFO_COND(debug, "[TIMING] Frontier computation: %.4f", timer.Toc());

    // Without a frontier, cannot do anything
    if (frontier.empty()) {
      ROS_WARN("Frontier cannot be computed. Exiting");

      visualize(costMap, traversible, occupied, skeletonPts, coordDrone, frontier,
                uavToSkeleton, alongSkeleton, skeletonToTarget, frontierToTarget,
                coordTarget, kResolution);
      delete mapPtr;
      return;
    }

    // Calculate distance map
    std::vector<cv::Point> targetPoints{coordTarget};
    costMap =
        computeCostMap(frontier, traversible, occupiedSafe, targetPoints, coordDrone);

    ROS_INFO_COND(debug, "[TIMING] Cost map computation: %.4f", timer.Toc());

    // If costmap computation was successful, compute a path from frontier to the
    // target otherwise, use frontier point in the middle as a placeholder, and
    // return empty path
    if (costMap.empty()) {
      ROS_ERROR("Cost map empty. Investigate");
      visualize(costMap, traversible, occupied, skeletonPts, coordDrone, frontier,
                uavToSkeleton, alongSkeleton, skeletonToTarget, frontierToTarget,
                coordTarget, kResolution);
      delete mapPtr;
      return;
    }

    // Select a default frontier point
    cv::Point selectedFrontierPt = frontier[frontier.size() / 2];

    // Find the frontier point with the shortest distance
    for (const cv::Point& pt : frontier) {
      if (costMap.at<uint16_t>(pt.y, pt.x) <
          costMap.at<uint16_t>(selectedFrontierPt.y, selectedFrontierPt.x)) {
        selectedFrontierPt = pt;
      }
    }

    // Compute path
    frontierToTarget = computeShortestPathFromCostmap(costMap, selectedFrontierPt);

    ROS_INFO_COND(debug, "[TIMING] Frontier shortest path computation: %.4f",
                  timer.Toc());

    // Path from skeleton to frontier
    cv::Point exitSkeletonPt =
        findClosestLoSSkeletonPoint(skeletonPts, selectedFrontierPt, traversible);
    skeletonToTarget =
        computeLineOfSightPath(traversible, exitSkeletonPt, selectedFrontierPt);
    ROS_INFO_COND(debug,
                  "[TIMING] Finding cloest LoS skeleton point to the frontier: %.4f",
                  timer.Toc());

    // Path along the skeleton between entryPoint and exitPoint
    alongSkeleton = findPathAlongSkeleton(skeletonPts, entrySkeletonPt, exitSkeletonPt);

    ROS_INFO_COND(debug, "[TIMING] Finding path along the skeleton: %.4f", timer.Toc());
  }

  ROS_INFO("AlongSkeleton: %d, skeletonToTarget: %d",
           static_cast<int>(alongSkeleton.size()),
           static_cast<int>(skeletonToTarget.size()));

  // Append the paths along and after skeleton
  path.insert(path.end(), alongSkeleton.begin(), alongSkeleton.end());
  path.insert(path.end(), skeletonToTarget.begin(), skeletonToTarget.end());

  // Convert path points to real coordinates and an angle
  // Each node is facing the next node in line
  std::vector<float> pathYaw, pathX, pathY;

  for (size_t i = 1; i < path.size(); ++i) {
    float xCurr = static_cast<float>(path[i - 1].x) * kResolution + mapXMin,
          yCurr = static_cast<float>(path[i - 1].y) * kResolution + mapYMin,
          xNext = static_cast<float>(path[i].x) * kResolution + mapXMin,
          yNext = static_cast<float>(path[i].y) * kResolution + mapYMin;

    pathX.push_back(xCurr);
    pathY.push_back(yCurr);
    pathYaw.push_back(std::atan2(yNext - yCurr, xNext - xCurr));
  }

  ROS_INFO_COND(debug, "[TIMING] Path composition: %.4f", timer.Toc());

  size_t N = pathX.size();

  // Filter x, y, and yaw separately with a FIR (order 10, cutoff 0.01)
  const std::vector<float> kernel{
      0.014548974194706, 0.030571249925298, 0.072545157760889, 0.124486576686153,
      0.166541934360414, 0.182612214145080, 0.166541934360414, 0.124486576686153,
      0.072545157760889, 0.030571249925298, 0.014548974194706,
  };

  std::vector<geometry_msgs::Pose> pathPoses;

  // Filter if the number of paths is longer than the filter kernel size
  if (N > kernel.size()) {
    // // Repeat front elements / last message N_rep times to retain initial /  final
    // // locations
    // const size_t N_rep = 5;
    // const std::vector<float> beginningX(N_rep, pathX[0]),
    //     endingX(N_rep, pathX[pathX.size() - 1]), beginningY(N_rep, pathY[0]),
    //     endingY(N_rep, pathY[pathY.size() - 1]), beginningYaw(N_rep, pathYaw[0]),
    //     endingYaw(N_rep, pathYaw[pathYaw.size() - 1]);

    // pathX.insert(pathX.begin(), beginningX.begin(), beginningX.end());
    // pathX.insert(pathX.end(), endingX.begin(), endingX.end());

    // pathY.insert(pathY.begin(), beginningY.begin(), beginningY.end());
    // pathY.insert(pathY.end(), endingY.begin(), endingY.end());

    // pathYaw.insert(pathYaw.begin(), beginningYaw.begin(), beginningYaw.end());
    // pathYaw.insert(pathYaw.end(), endingYaw.begin(), endingYaw.end());

    const size_t N_filt = N - kernel.size() + 1;  // + 2 * N_rep;
    pathPoses = std::vector<geometry_msgs::Pose>(N_filt);

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < N_filt; ++i) {
      // Filter sin & cos to avoid discontinuities
      float x_filt = 0., y_filt = 0., sin_filt = 0., cos_filt = 0.;

      for (size_t j = 0; j < kernel.size(); ++j) {
        x_filt += kernel[j] * pathX[i + j];
        y_filt += kernel[j] * pathY[i + j];
        float yaw = pathYaw[i + j];
        sin_filt += kernel[j] * std::sin(yaw);
        cos_filt += kernel[j] * std::cos(yaw);
      }

      float t_filt = std::atan2(sin_filt, cos_filt);

      pathPoses[i].position.x = x_filt;
      pathPoses[i].position.y = y_filt;
      pathPoses[i].position.z = kNavigationHeight;
      pathPoses[i].orientation.x = 0.0;
      pathPoses[i].orientation.y = 0.0;
      pathPoses[i].orientation.z = std::sin(t_filt / 2.);
      pathPoses[i].orientation.w = std::cos(t_filt / 2.);
    }

    ROS_INFO_COND(debug, "[TIMING] Path filtering: %.4f", timer.Toc());
  } else {
    pathPoses = std::vector<geometry_msgs::Pose>(N);
    for (size_t i = 0; i < N; ++i) {
      pathPoses[i].position.x = pathX[i];
      pathPoses[i].position.y = pathY[i];
      pathPoses[i].position.z = kNavigationHeight;
      pathPoses[i].orientation.x = 0.0;
      pathPoses[i].orientation.y = 0.0;
      pathPoses[i].orientation.z = std::sin(pathYaw[i] / 2.);
      pathPoses[i].orientation.w = std::cos(pathYaw[i] / 2.);
    }
  }
  // Publish message
  static uint32_t seq = 0;
  geometry_msgs::PoseArray uavToFrontier;
  uavToFrontier.header.seq = seq;
  uavToFrontier.header.stamp = ros::Time::now();
  uavToFrontier.header.frame_id = "map";
  uavToFrontier.poses = pathPoses;
  pathPub.publish(uavToFrontier);

  visualize(costMap, traversible, occupied, skeletonPts, coordDrone, frontier,
            uavToSkeleton, alongSkeleton, skeletonToTarget, frontierToTarget,
            coordTarget, kResolution);

  // Free map pointers
  delete mapPtr;

  ROS_INFO(" ##### Exiting pathfinder #####");

  return;
}

int main(int argc, char** argv) {
  const std::string nodeName = "pathfinding_node";
  ros::init(argc, argv, nodeName);
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  tf2_ros::TransformListener tfListener(tfBuffer);
  image_transport::ImageTransport imageTransport(nh);
  debugVis = imageTransport.advertise("debug_vis", 1);
  pathPub = nh.advertise<geometry_msgs::PoseArray>("navigation_path", 1, true);

  ros::Subscriber octomapMsgSubscriber =
      nh.subscribe("/rtabmap/octomap_binary", 1, octomapCallback);

  ros::spin();

  return 0;
}
