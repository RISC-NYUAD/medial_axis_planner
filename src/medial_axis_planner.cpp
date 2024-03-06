#include "medial_axis_planner/medial_axis_planner.hpp"
#include "nav2_util/node_utils.hpp"
#include "timer.hpp"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

namespace medial_axis_planner {

MedialAxis::MedialAxis() : tf_(nullptr), costmap_(nullptr) {}
MedialAxis::~MedialAxis() {
  RCLCPP_INFO_STREAM(logger_,
                     "Destroying plugin " << name_ << " of type MedialAxis");
}

void MedialAxis::configure(
    const rclcpp_lifecycle::LifecycleNode::WeakPtr &parent, std::string name,
    std::shared_ptr<tf2_ros::Buffer> tf,
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) {

  node_ = parent.lock();
  clock_ = node_->get_clock();
  logger_ = node_->get_logger();

  RCLCPP_INFO_STREAM(logger_,
                     "Configuring plugin " << name_ << " of type MedialAxis");

  tf_ = tf;
  name_ = name;
  costmap_ = costmap_ros->getCostmap();
  footprint_ = costmap_ros->getUnpaddedRobotFootprint();
  global_frame_ = costmap_ros->getGlobalFrameID();

  // Publisher for debug image
  debug_publisher_ =
      node_->create_publisher<sensor_msgs::msg::Image>("planner_visual", 10);

  // // Parameter initialization
  // nav2_util::declare_parameter_if_not_declared(
  //     node_, name_ + ".interpolation_resolution",
  //     rclcpp::ParameterValue(0.1));
  // node_->get_parameter(name_ + ".interpolation_resolution",
  //                      interpolation_resolution_);
  // RCLCPP_INFO(logger_, "MedialAxis configuration complete");

  // Populate lookup tables
  // Anything up until MAX_NON_OBSTACLE (252) can be traversed
  // Check "nav2_costmap_2d/cost_values.hpp" for values
  free_lut_ = cv::Mat(1, 256, CV_8U, cv::Scalar(0));
  free_lut_.at<uint8_t>(0, nav2_costmap_2d::FREE_SPACE) = 255;

  unknown_lut_ = cv::Mat(1, 256, CV_8U, cv::Scalar(0));
  unknown_lut_.at<uint8_t>(0, nav2_costmap_2d::NO_INFORMATION) = 255;

  occupied_lut_ = cv::Mat(1, 256, CV_8U, cv::Scalar(0));
  occupied_lut_.at<uint8_t>(0, nav2_costmap_2d::LETHAL_OBSTACLE) = 255;
  // occupied_lut_.at<uint8_t>(0, nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE)
  // = 255;

  // Anything between free and MAX_NON_OBSTACLE is dangerous to approach, but
  // they are still valid
  danger_lut_ = cv::Mat(1, 256, CV_8U, cv::Scalar(0));
  for (size_t i = 1; i <= nav2_costmap_2d::MAX_NON_OBSTACLE; ++i)
    danger_lut_.at<uint8_t>(0, i) = 255;

  // Remap distances to logical values. By default, NO_INFORMATION is 255, and
  // LETHAL_OBSTACLE is 254. We allow traversing unknown, so we should treat it
  // (and visualize it) as somewhat free
  costmap_lut_ = cv::Mat(1, 256, CV_8U);
  for (size_t i = 0; i < 256; ++i) {
    costmap_lut_.at<uint8_t>(0, i) = i;
  }
  costmap_lut_.at<uint8_t>(0, nav2_costmap_2d::NO_INFORMATION) =
      nav2_costmap_2d::MAX_NON_OBSTACLE;
}

void MedialAxis::cleanup() {
  RCLCPP_INFO_STREAM(logger_,
                     "Cleaning up plugin " << name_ << " of type MedialAxis");
}
void MedialAxis::activate() {
  RCLCPP_INFO_STREAM(logger_,
                     "Activating plugin " << name_ << " of type MedialAxis");
}
void MedialAxis::deactivate() {
  RCLCPP_INFO_STREAM(logger_,
                     "Deactivating plugin " << name_ << " of type MedialAxis");
}

nav_msgs::msg::Path
MedialAxis::createPlan(const geometry_msgs::msg::PoseStamped &start,
                       const geometry_msgs::msg::PoseStamped &goal) {

  static Timer t;

  RCLCPP_INFO(logger_, "New path requested from MedialAxis planner");
  const double dx = goal.pose.position.x - start.pose.position.x,
               dy = goal.pose.position.y - start.pose.position.y;

  const size_t size_x = costmap_->getSizeInCellsX(),
               size_y = costmap_->getSizeInCellsY();

  // For OpenCV mat converted in below method, x right, y down (z inwards)
  // Use cv::flip(src, src, 0) for z upwards
  t.tic();
  cv::Mat costmap_raw(size_y, size_x, CV_8UC1, costmap_->getCharMap());
  RCLCPP_INFO_STREAM(logger_, "[TIMING] costmap2d->Mat: " << t.toc());

  // Lookup tables to extract individual layers
  t.tic();
  cv::Mat occupied, free, unknown, dangerous, costmap;
  cv::LUT(costmap_raw, occupied_lut_, occupied);
  cv::LUT(costmap_raw, free_lut_, free);
  cv::LUT(costmap_raw, unknown_lut_, unknown);
  cv::LUT(costmap_raw, danger_lut_, dangerous);
  cv::LUT(costmap_raw, costmap_lut_, costmap);
  RCLCPP_INFO_STREAM(logger_, "[TIMING] Extract zones: " << t.toc());

  // Extract start and end coordinates
  cv::Point start_coords, goal_coords;
  {
    unsigned int cx, cy;
    costmap_->worldToMap(start.pose.position.x, start.pose.position.y, cx, cy);
    start_coords.x = cx;
    start_coords.y = cy;

    costmap_->worldToMap(goal.pose.position.x, goal.pose.position.y, cx, cy);
    goal_coords.x = cx;
    goal_coords.y = cy;

    // TODO: Enforce boundary checks
  }

  // FIXME: make including unknown cells optional
  t.tic();
  cv::Mat traversible = free + dangerous + unknown;
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::morphologyEx(traversible, traversible, cv::MORPH_CLOSE, kernel);
  cv::morphologyEx(traversible, traversible, cv::MORPH_OPEN, kernel);
  RCLCPP_INFO_STREAM(logger_, "[TIMING] Opening and closing: " << t.toc());

  // // Verify that the goal is traversible / not in lethal obstacle zone
  // if (traversible.at<uint8_t>(goal_coords) == 0) {
  //   RCLCPP_ERROR(logger_, "Goal location is not traversible");
  // }

  // // Extract the region that contains the starting (current) position
  // RegionContours region_containing_start_pos =
  //     extractRegionContainingCoordinate(traversible, start_coords, true);

  // t.tic();
  // {
  //   cv::Mat traversible_region_mask(traversible.size(), CV_8UC1, cv::Scalar(0));
  //   cv::fillPoly(traversible_region_mask,
  //                {region_containing_start_pos.external}, cv::Scalar(1));
  //   cv::fillPoly(traversible_region_mask, region_containing_start_pos.internal,
  //                cv::Scalar(0));
  //   traversible = traversible.mul(traversible_region_mask);
  // }
  // RCLCPP_INFO_STREAM(logger_, "[TIMING] Masking: " << t.toc());

  // Prevent the border from being traversible. Adds weird axis points
  {
    cv::Mat border_mask(traversible.size(), CV_8UC1, cv::Scalar(0));
    std::vector<cv::Point> vertices{cv::Point(1, 1), cv::Point(size_x - 2, 1),
                                    cv::Point(size_x - 2, size_y - 2),
                                    cv::Point(1, size_y - 2)};
    cv::fillConvexPoly(border_mask, vertices, cv::Scalar(1));
    traversible = traversible.mul(border_mask);
  }

  // Declare some variables in advance to place them in correct scope
  std::vector<cv::Point> medial_axis_pts, path_map;
  nav_msgs::msg::Path path;

  // Check if a line-of-sight path is possible
  double start_to_goal_distance = std::hypot(dx, dy);
  if (start_to_goal_distance < 2.0)
    path_map = computeLineOfSightPath(costmap, start_coords, goal_coords);

  if (path_map.empty()) {

    // Extract medial axis (i.e. skeleton) coordinates
    t.tic();
    cv::Mat medial_axis;
    cv::ximgproc::thinning(traversible, medial_axis);
    cv::findNonZero(medial_axis, medial_axis_pts);
    RCLCPP_INFO_STREAM(logger_, "[TIMING] Medial axis: " << t.toc());

    // Shortest/Cheapest path from start to skeleton
    std::vector<cv::Point> medial_axis_entry =
        findClosestLineOfSightFrom(medial_axis_pts, start_coords, costmap);
    if (medial_axis_entry.empty()) {
      RCLCPP_WARN(logger_, "Cannot find a valid entry path to the medial axis");
    }
    // Reverse to direct from start to medial axis
    std::reverse(medial_axis_entry.begin(), medial_axis_entry.end());

    // Shortest/Cheapest path from skeleton to goal
    std::vector<cv::Point> medial_axis_exit =
        findClosestLineOfSightFrom(medial_axis_pts, goal_coords, costmap);
    if (medial_axis_exit.empty()) {
      RCLCPP_WARN(logger_,
                  "Cannot find a valid exit path from the medial axis");
    }

    // Path along the medial axis itself
    if (!medial_axis_entry.empty() and !medial_axis_exit.empty()) {
      std::vector<cv::Point> path_along_medial_axis =
          findPathAlongMedialAxis(medial_axis_pts, medial_axis_entry.back(),
                                  medial_axis_exit.at(0), costmap);
      // If path along the medial axis is empty, path is invalid
      if (!path_along_medial_axis.empty()) {

        // Add up until the last item to not double count some points
        path_map.insert(path_map.end(), medial_axis_entry.begin(),
                        medial_axis_entry.end() - 1);
        path_map.insert(path_map.end(), path_along_medial_axis.begin(),
                        path_along_medial_axis.end() - 1);
        path_map.insert(path_map.end(), medial_axis_exit.begin(),
                        medial_axis_exit.end());
      } else {
        RCLCPP_WARN(
            logger_,
            "Cannot find a valid path along the medial axis. Disconnected?");
      }
    }
  }

  // Assign angles based on next point in line
  const size_t N = path_map.size();
  std::vector<double> path_yaw;

  if (N == 0) {
    RCLCPP_WARN(logger_, "Could not compute path");
  } else {
    path_yaw.resize(N);
    for (size_t i = 0; i < N - 1; ++i) {
      const cv::Point &current = path_map.at(i), &next = path_map.at(i + 1);
      path_yaw.at(i) = std::atan2(next.y - current.y, next.x - current.x);
    }
    // FIXME: Where should we get the last yaw from?
    path_yaw.at(N - 1) = path_yaw.at(N - 2);

    // Convert from map to world, append the final goal path
    path = convertMapPathToRealPath(path_map, path_yaw, start.header);
    path.poses.push_back(goal);
  }

  RCLCPP_INFO(logger_, "Path planning completed");

  // Visualization
  cv::Mat zeros(size_y, size_x, CV_8UC1, cv::Scalar(0));
  cv::Mat costmap_inverse(costmap.size(), CV_8UC1, cv::Scalar(255));
  costmap_inverse = costmap_inverse - costmap;

  // std::vector<cv::Mat> channels{zeros, costmap_inverse, occupied};
  std::vector<cv::Mat> channels{zeros, traversible, occupied};
  cv::Mat visual;
  cv::merge(channels, visual);

  // Mark the medial axis
  for (const auto &pt : medial_axis_pts) {
    cv::circle(visual, pt, 0, cv::Scalar(100, 100, 100), 1);
  }

  // Mark the path
  for (const auto &pt : path_map) {
    cv::circle(visual, pt, 0, cv::Scalar(200, 200, 200), 1);
  }

  // Mark the start and end coordinates
  cv::circle(visual, start_coords, 1, cv::Scalar(255, 0, 255), 1);
  cv::circle(visual, goal_coords, 1, cv::Scalar(155, 155, 0), 1);

  // // Draw some of the boxes
  // for (size_t i = 0; i < N; i = i + 10) {

  //   const auto &position = path_map.at(i);
  //   double yaw = path_yaw.at(i);
  //   double cost = computePoseCost(position, yaw, costmap);

  //   // Find out the oriented footprint coordinates
  //   double wx, wy;
  //   costmap_->mapToWorld(position.x, position.y, wx, wy);

  //   std::vector<geometry_msgs::msg::Point> footprint_rotated(4);
  //   std::vector<cv::Point> footprint_map(4);

  //   const double cos = std::cos(yaw), sin = std::sin(yaw);

  //   for (size_t j = 0; j < 4; ++j) {
  //     const geometry_msgs::msg::Point &corner = footprint_.at(j);
  //     geometry_msgs::msg::Point &corner_rotated = footprint_rotated.at(j);

  //     corner_rotated.x = wx + cos * corner.x - sin * corner.y;
  //     corner_rotated.y = wy + sin * corner.x + cos * corner.y;

  //     unsigned int cx, cy;
  //     costmap_->worldToMap(corner_rotated.x, corner_rotated.y, cx, cy);
  //     footprint_map.at(j).x = cx;
  //     footprint_map.at(j).y = cy;
  //   }

  //   for (size_t j = 0; j < 4; ++j) {
  //     const auto &start_line = footprint_map.at(j),
  //                &end_line = footprint_map.at((j + 1) % 4);
  //     cv::line(visual, start_line, end_line, cv::Scalar(255, 0, 0));
  //   }
  // }

  cv::flip(visual, visual, 0);

  std_msgs::msg::Header header;
  sensor_msgs::msg::Image::SharedPtr visual_msg =
      cv_bridge::CvImage(header, "bgr8", visual).toImageMsg();
  debug_publisher_->publish(*visual_msg.get());

  return path;
}

RegionContours
MedialAxis::extractRegionContainingCoordinate(const cv::Mat &traversible,
                                              const cv::Point &coordinate,
                                              bool simplify) const {
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;

  if (simplify)
    cv::findContours(traversible, contours, hierarchy, cv::RETR_TREE,
                     cv::CHAIN_APPROX_SIMPLE);
  else
    cv::findContours(traversible, contours, hierarchy, cv::RETR_TREE,
                     cv::CHAIN_APPROX_NONE);

  // Select the contour the drone is in
  RegionContours relevant_contour;
  relevant_contour.valid = false;
  bool contour_is_set = false;

  for (size_t i = 0; i < contours.size(); ++i) {
    bool coordinate_in_hole = false;

    const std::vector<cv::Point> &contour = contours[i];

    // Check if the coordinate is inside the outermost boundaries
    if (cv::pointPolygonTest(contour, coordinate, false) > 0) {
      // Check if the coordinate is inside the holes of the contour
      std::vector<std::vector<cv::Point>> children_contours;
      for (size_t j = 0; j < contours.size(); ++j)
        if (hierarchy[j][3] == static_cast<int>(i))
          // Parent is the current contour
          children_contours.push_back(contours[j]);

      for (const std::vector<cv::Point> &child_contour : children_contours) {
        // If the coordinate is inside a hole, then this is not the contour we
        // are looking for
        if (cv::pointPolygonTest(child_contour, coordinate, false) > 0)
          coordinate_in_hole = true;
        break;
      }

      if (!coordinate_in_hole) {
        relevant_contour.external = contour;
        relevant_contour.internal = children_contours;
        relevant_contour.valid = true;
        contour_is_set = true;
        break;
      }
    }

    if (contour_is_set)
      break;
  }

  if (!relevant_contour.valid) {
    RCLCPP_WARN(logger_, "Contour cannot be set properly. Investigate");
  }

  return relevant_contour;
}

std::vector<cv::Point> MedialAxis::computeLineOfSightPath(
    const cv::Mat &region, const cv::Point &start, const cv::Point &end) const {
  std::vector<cv::Point> retval;
  cv::LineIterator it(region, start, end, 8);

  for (int i = 0; i < it.count; i++, ++it) {
    // nav2_costmap_2d::MAX_NON_OBSTACLE = 252
    if (region.at<uint8_t>(it.pos()) > 252) {
      return std::vector<cv::Point>();
    } else {
      retval.push_back(it.pos());
    }
  }
  return retval;
}

std::vector<cv::Point>
MedialAxis::findClosestLineOfSightFrom(const std::vector<cv::Point> &candidates,
                                       const cv::Point &target,
                                       const cv::Mat &costmap) const {
  float current_distance = std::numeric_limits<float>::max();

  std::vector<cv::Point> closest_path;

  for (const cv::Point &pt : candidates) {
    cv::Point diff = pt - target;
    float dist = std::hypot(diff.x, diff.y);

    if (dist < current_distance) {
      // LoS needed only if the distance is smaller
      std::vector<cv::Point> line_of_sight_path =
          computeLineOfSightPath(costmap, pt, target);

      // If LoS path is empty, then there is no path
      if (line_of_sight_path.empty()) {
        continue;
      } else {
        current_distance = dist;
        closest_path = line_of_sight_path;
      }
    }
  }

  return closest_path;
}

std::vector<cv::Point> MedialAxis::findCheapestLineOfSightFrom(
    const std::vector<cv::Point> &candidates, const cv::Point &target,
    const cv::Mat &costmap) const {

  size_t current_cost = std::numeric_limits<size_t>::max();

  std::vector<cv::Point> selected_path;

  // Cost is comprised of L1 distance and the sum of the costs of poses at the
  // start, end, and midpoint. Assumed constant yaw from start to end
  for (const cv::Point &pt : candidates) {
    cv::Point diff = pt - target, midpoint;
    midpoint.x = (pt.x + target.x) / 2;
    midpoint.y = (pt.y + target.y) / 2;

    const double yaw = std::atan2(diff.y, diff.x);

    // RCLCPP_INFO_STREAM(
    //     logger_, "Start: (" << pt.x << ", " << pt.y << ") Target: (" <<
    //     target.x
    //                         << ", " << target.y << ") Midpoint: (" <<
    //                         midpoint.x
    //                         << ", " << midpoint.y << ")");

    double start_cost = computePoseCost(pt, yaw, costmap),
           end_cost = computePoseCost(target, yaw, costmap),
           midpoint_cost = computePoseCost(midpoint, yaw, costmap);

    size_t total_cost =
        std::abs(diff.x) + std::abs(diff.y) + static_cast<size_t>(start_cost) +
        static_cast<size_t>(midpoint_cost) + static_cast<size_t>(end_cost);

    if (total_cost < current_cost) {
      // LoS needed only if the distance is smaller
      std::vector<cv::Point> line_of_sight_path =
          computeLineOfSightPath(costmap, pt, target);

      // If LoS path is empty, then there is no path
      if (line_of_sight_path.empty()) {
        continue;
      } else {
        current_cost = total_cost;
        selected_path = line_of_sight_path;
      }
    }
  }

  return selected_path;
}

std::vector<cv::Point> MedialAxis::findPathAlongMedialAxis(
    const std::vector<cv::Point> &medial_axis_pts, const cv::Point &start,
    const cv::Point &end, const cv::Mat &costmap) const {

  // Struct to hold point and a distance
  struct PtWithDistance {
    size_t idx;
    cv::Point pt;
    size_t dist;
  };

  // Comparator reversed for higher priority on small distances
  struct PtWithDistanceComparator {
    bool operator()(const PtWithDistance &lhs,
                    const PtWithDistance &rhs) const {
      return lhs.dist > rhs.dist;
    }
  };

  // Comparator for cv::Point types
  struct cvPointComparator {
    bool operator()(const cv::Point &lhs, const cv::Point &rhs) const {
      if (lhs.x == rhs.x)
        return lhs.y < rhs.y;
      return lhs.x < rhs.x;
    }
  };

  // If end and start are the same, simply return one of them
  if (start == end)
    return std::vector<cv::Point>{start};

  // Keep track of distance and visit status to deal with updating requirements
  size_t N = medial_axis_pts.size();
  // std::vector<uint16_t> nodeDistances(N,
  // std::numeric_limits<uint16_t>::max());
  std::vector<size_t> nodeDistances(N, std::numeric_limits<size_t>::max());

  std::vector<bool> visited(N, false);

  // Previous path history to trace the path from one node to other
  std::vector<size_t> pathPrev(N, std::numeric_limits<size_t>::max());

  // Assign a mapping from point to their index
  std::map<cv::Point, size_t, cvPointComparator> ptToIdx;
  for (size_t i = 0; i < N; ++i) {
    ptToIdx[medial_axis_pts[i]] = i;
  }

  // Ensure that the start and end are actually inside the skeleton
  std::map<cv::Point, size_t>::iterator itStart, itEnd;
  itStart = ptToIdx.find(start);
  itEnd = ptToIdx.find(end);
  if (itStart == ptToIdx.end() || itEnd == ptToIdx.end()) {
    RCLCPP_ERROR(
        logger_,
        "Start or end point is not part of the medial axis. Investigate");
    return std::vector<cv::Point>();
  }

  // Distance-based priority queue
  std::priority_queue<PtWithDistance, std::vector<PtWithDistance>,
                      PtWithDistanceComparator>
      pq;

  // Starting point has 0 distance
  nodeDistances[ptToIdx[start]] = 0;
  for (size_t i = 0; i < N; ++i)
    pq.push(PtWithDistance{i, medial_axis_pts[i], nodeDistances[i]});

  // Mark start to be visited
  visited[ptToIdx[start]] = true;

  while (!pq.empty()) {
    PtWithDistance curr = pq.top();
    pq.pop();

    // Ensure distance is up to date
    if (curr.dist > nodeDistances[curr.idx])
      continue;

    // Terminate if the end node is reached
    // NOTE: Does not work if the skeleton is comprised of 2 components. Need
    // all paths
    // if (curr.pt == end) break;

    visited[curr.idx] = true;

    std::vector<cv::Point> neighborhood = computeNeighbors(curr.pt);

    for (const cv::Point &neighbor : neighborhood) {
      // Skip if not in skeleton
      if (ptToIdx.find(neighbor) == ptToIdx.end())
        continue;

      size_t neighborIdx = ptToIdx[neighbor];

      // Skip if visited
      if (visited[neighborIdx])
        continue;

      // Distance is step + cost of the cell, without scaling
      // This way, the aim is to avoid dangerous cells as much as possible
      size_t cost = static_cast<size_t>(costmap.at<uint8_t>(neighbor)),
             multiplier = 1;

      // Blow up lethal obstacle + its inflated border costs
      // Anything above nav2_costmap_2d::MAX_NON_OBSTACLE (= 252)
      if (cost > 252)
        multiplier = 100;

      size_t newDist = curr.dist + 1 + cost * multiplier;

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
    RCLCPP_WARN(logger_, "No path exists between given start and end points");
    return std::vector<cv::Point>();
  }

  // Calculate path from end to start
  std::vector<cv::Point> path;
  size_t curIdx = endIdx;
  while (curIdx != startIdx) {
    path.push_back(medial_axis_pts[curIdx]);
    curIdx = pathPrev[curIdx];
  }

  // Invert the path to orient from start to end
  std::reverse(path.begin(), path.end());
  return path;
}

std::vector<cv::Point> MedialAxis::computeNeighbors(const cv::Point &pt) const {
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

nav_msgs::msg::Path MedialAxis::convertMapPathToRealPath(
    const std::vector<cv::Point> &path_map, const std::vector<double> &path_yaw,
    const std_msgs::msg::Header &header) const {

  nav_msgs::msg::Path path_world;
  path_world.header = header;

  const size_t N = path_map.size();
  path_world.poses.resize(N);

  // Iterating up to N-1, so that the last entry exists
  for (size_t i = 0; i < N; ++i) {
    const cv::Point &pt = path_map.at(i);
    auto &pose = path_world.poses.at(i);
    pose.header = header;

    double x, y;
    costmap_->mapToWorld(pt.x, pt.y, x, y);
    pose.pose.position.x = x;
    pose.pose.position.y = y;

    double yaw = path_yaw.at(i);
    pose.pose.orientation.x = 0.0;
    pose.pose.orientation.y = 0.0;
    pose.pose.orientation.z = std::sin(yaw / 2.);
    pose.pose.orientation.w = std::cos(yaw / 2.);
  }

  return path_world;
}

double MedialAxis::computePoseCost(cv::Point position, double yaw,
                                   const cv::Mat &costmap) const {
  // FIXME: Compute more optimally (integral image-like)
  // Compute corners based on yaw and footprint to create a mask
  double wx, wy;
  costmap_->mapToWorld(position.x, position.y, wx, wy);

  std::vector<geometry_msgs::msg::Point> footprint_rotated(4);
  std::vector<cv::Point> footprint_map(4);

  const double cos = std::cos(yaw), sin = std::sin(yaw);

  // RCLCPP_INFO_STREAM(logger_, "Position: (" << position.x << ", " <<
  // position.y
  //                                           << ") Yaw: " << yaw);

  for (size_t i = 0; i < 4; ++i) {
    const geometry_msgs::msg::Point &corner = footprint_.at(i);
    geometry_msgs::msg::Point &corner_rotated = footprint_rotated.at(i);

    corner_rotated.x = wx + cos * corner.x - sin * corner.y;
    corner_rotated.y = wy + sin * corner.x + cos * corner.y;

    int cx, cy;
    costmap_->worldToMapNoBounds(corner_rotated.x, corner_rotated.y, cx, cy);
    footprint_map.at(i).x = cx;
    footprint_map.at(i).y = cy;
  }

  // Restrict calculations to ROI of the footprint to save time
  const auto &pt = footprint_map.at(0);
  int min_x = pt.x, min_y = pt.y, max_x = pt.x, max_y = pt.y;
  for (size_t i = 1; i < 4; ++i) {
    min_x = std::max(std::min(min_x, footprint_map.at(i).x), 0);
    min_y = std::max(std::min(min_y, footprint_map.at(i).y), 0);

    max_x = std::min(std::max(max_x, footprint_map.at(i).x), costmap.cols - 1);
    max_y = std::min(std::max(max_y, footprint_map.at(i).y), costmap.rows - 1);
  }

  // RCLCPP_INFO_STREAM(logger_, "Position: (" << position.x << ", " <<
  // position.y
  //                                           << ") Yaw: " << yaw << " x: ["
  //                                           << min_x << ", " << max_x
  //                                           << "] y: [" << min_y << ", "
  //                                           << max_y << "]");

  std::vector<cv::Point> footprint_roi(4);
  for (size_t i = 0; i < 4; ++i) {
    const auto &pt = footprint_map.at(i);
    footprint_roi.at(i).x = pt.x - min_x;
    footprint_roi.at(i).y = pt.y - min_y;
  }

  if (min_x < 0 or min_y < 0 or max_x >= costmap.cols or
      max_y >= costmap.rows) {
    RCLCPP_WARN_STREAM(logger_, "ROI out of bounds: x: ["
                                    << min_x << ", " << max_x << "] y: ["
                                    << min_y << ", " << max_y << "]");
  }

  // By default, ROI image shares data with the underlying Mat. Clone ROI to
  // prevent overwrite
  cv::Rect roi(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
  cv::Mat costmap_roi_raw(costmap, roi);
  cv::Mat costmap_roi = costmap_roi_raw.clone();

  cv::Mat mask(costmap_roi.size(), CV_8UC1, cv::Scalar(0));
  cv::fillConvexPoly(mask, footprint_roi, cv::Scalar(1));
  costmap_roi = costmap_roi.mul(mask);
  return cv::sum(costmap_roi)[0];
}

} // namespace medial_axis_planner

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(medial_axis_planner::MedialAxis,
                       nav2_core::GlobalPlanner)
