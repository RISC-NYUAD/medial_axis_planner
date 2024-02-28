#include "medial_axis_planner/medial_axis_planner.hpp"
#include "nav2_util/node_utils.hpp"
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

  tf_ = tf;
  name_ = name;
  costmap_ = costmap_ros->getCostmap();
  global_frame_ = costmap_ros->getGlobalFrameID();

  node_ = parent.lock();
  clock_ = node_->get_clock();
  logger_ = node_->get_logger();

  RCLCPP_INFO_STREAM(logger_,
                     "Configuring plugin " << name_ << " of type MedialAxis");

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

  // Anything between free and MAX_NON_OBSTACLE is dangerous to approach
  danger_lut_ = cv::Mat(1, 256, CV_8U, cv::Scalar(0));
  for (size_t i = 1; i <= nav2_costmap_2d::MAX_NON_OBSTACLE; ++i)
    danger_lut_.at<uint8_t>(0, i) = 255;
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

  const size_t size_x = costmap_->getSizeInCellsX(),
               size_y = costmap_->getSizeInCellsY();

  // For OpenCV mat converted in below method, x right, y down (z inwards)
  // Use cv::flip(src, src, 0) for z upwards
  cv::Mat costmap_raw(size_y, size_x, CV_8UC1, costmap_->getCharMap());

  // Lookup tables to extract individual layers
  cv::Mat occupied, free, unknown, dangerous;
  cv::LUT(costmap_raw, occupied_lut_, occupied);
  cv::LUT(costmap_raw, free_lut_, free);
  cv::LUT(costmap_raw, unknown_lut_, unknown);
  cv::LUT(costmap_raw, danger_lut_, dangerous);

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
  cv::Mat traversible = free + dangerous + unknown;
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::morphologyEx(traversible, traversible, cv::MORPH_CLOSE, kernel);
  cv::morphologyEx(traversible, traversible, cv::MORPH_OPEN, kernel);

  // Extract the region that contains the starting (current) position
  RegionContours region_containing_start_pos =
      extractRegionContainingCoordinate(traversible, start_coords, true);

  {
    cv::Mat traversible_region_mask(traversible.size(), CV_8UC1, cv::Scalar(0));
    cv::fillPoly(traversible_region_mask,
                 {region_containing_start_pos.external}, cv::Scalar(1));
    cv::fillPoly(traversible_region_mask, region_containing_start_pos.internal,
                 cv::Scalar(0));
    traversible = traversible.mul(traversible_region_mask);
  }

  // Prevent the border from being traversible. Adds weird axis points
  {
    cv::Mat border_mask(traversible.size(), CV_8UC1, cv::Scalar(0));
    std::vector<cv::Point> vertices{cv::Point(1, 1), cv::Point(size_x - 2, 1),
                                    cv::Point(size_x - 2, size_y - 2),
                                    cv::Point(1, size_y - 2)};
    cv::fillConvexPoly(border_mask, vertices, cv::Scalar(1));
    traversible = traversible.mul(border_mask);
  }

  // Extract medial axis (i.e. skeleton) coordinates
  cv::Mat medial_axis;
  cv::ximgproc::thinning(traversible, medial_axis);
  std::vector<cv::Point> medial_axis_pts;
  cv::findNonZero(medial_axis, medial_axis_pts);

  // Shortest path from start to skeleton
  cv::Point medial_axis_entry =
      findClosesetLineofSightFrom(medial_axis_pts, start_coords, traversible);

  // Shortest path from skeleton to goal
  // TODO: Use costmap for selection?
  cv::Point medial_axis_exit =
      findClosesetLineofSightFrom(medial_axis_pts, goal_coords, traversible);

  // Path along the medial axis itself
  std::vector<cv::Point> path_along_medial_axis = findPathAlongMedialAxis(
      medial_axis_pts, medial_axis_entry, medial_axis_exit);

  // Construct path
  std::vector<cv::Point> path_map;
  {
    std::vector<cv::Point> start_to_medial_axis =
        computeLineOfSightPath(traversible, start_coords, medial_axis_entry);
    std::vector<cv::Point> medial_axis_to_goal =
        computeLineOfSightPath(traversible, medial_axis_exit, goal_coords);

    path_map.insert(path_map.end(), start_to_medial_axis.begin(),
                    start_to_medial_axis.end());
    path_map.insert(path_map.end(), path_along_medial_axis.begin(),
                    path_along_medial_axis.end());
    path_map.insert(path_map.end(), medial_axis_to_goal.begin(),
                    medial_axis_to_goal.end());
  }

  // Visualization
  cv::Mat zeros(size_y, size_x, CV_8UC1, cv::Scalar(0));
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

  cv::flip(visual, visual, 0);

  std_msgs::msg::Header header;
  sensor_msgs::msg::Image::SharedPtr visual_msg =
      cv_bridge::CvImage(header, "bgr8", visual).toImageMsg();
  debug_publisher_->publish(*visual_msg.get());

  nav_msgs::msg::Path path = convertMapPathToRealPath(path_map, start.header);

  // Append the goal
  path.poses.push_back(goal);

  return path;
}

RegionContours MedialAxis::extractRegionContainingCoordinate(
    const cv::Mat &traversible, const cv::Point &coordinate, bool simplify) {
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

// Give coordinates of the closest line-of-sight point from the set
cv::Point MedialAxis::findClosesetLineofSightFrom(
    const std::vector<cv::Point> &candidates, const cv::Point &target,
    const cv::Mat &valid_region) {
  float current_distance = std::numeric_limits<float>::max(),
        backup_distance = std::numeric_limits<float>::max();

  cv::Point closest_point, backup_point;
  bool closest_point_found = false;

  for (const cv::Point &pt : candidates) {
    cv::Point diff = pt - target;
    float dist = std::hypot(diff.x, diff.y);

    // No LoS check / closest point
    if (dist < backup_distance) {
      backup_point = pt;
      backup_distance = dist;
    }

    if (dist < current_distance) {
      // LoS needed only if the distance is smaller
      std::vector<cv::Point> line_of_sight_path =
          computeLineOfSightPath(valid_region, pt, target);

      // If LoS path is empty, then there is no path
      if (line_of_sight_path.empty()) {
        continue;
      } else {
        current_distance = dist;
        closest_point_found = true;
        closest_point = pt;
      }
    }
  }

  if (closest_point_found == false) {
    // FIXME: Assigning a point here so that the returned point is still from
    // among the candidates when NO LOS PATH EXISTS. THIS MAY BE DANGEROUS
    RCLCPP_WARN(
        logger_,
        "No line-of-sight skeleton point exists. Returning closest point");
    return backup_point;
  }

  return closest_point;
}
std::vector<cv::Point> MedialAxis::computeLineOfSightPath(
    const cv::Mat &region, const cv::Point &start, const cv::Point &end) {
  std::vector<cv::Point> retval;
  cv::LineIterator it(region, start, end, 8);

  // Starting from 1 to exclude the starting point
  ++it;
  for (int i = 1; i < it.count; i++, ++it) {
    if (region.at<uint8_t>(it.pos()) != 255) {
      return std::vector<cv::Point>();
    } else {
      retval.push_back(it.pos());
    }
  }
  return retval;
}

std::vector<cv::Point> MedialAxis::findPathAlongMedialAxis(
    const std::vector<cv::Point> &medial_axis_pts, const cv::Point &start,
    const cv::Point &end) {

  // Struct to hold point and a distance
  struct PtWithDistance {
    size_t idx;
    cv::Point pt;
    // uint16_t dist; // For l-inf
    float dist;
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
  std::vector<float> nodeDistances(N, std::numeric_limits<float>::max());

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

  // Start point has 0 distance
  nodeDistances[ptToIdx[start]] = 0;
  for (size_t i = 0; i < N; ++i)
    pq.push(PtWithDistance{i, medial_axis_pts[i], nodeDistances[i]});

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

    for (const cv::Point &neighbor : neighborhood) {
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

// Compute 8-neighbors of a given point
std::vector<cv::Point> MedialAxis::computeNeighbors(const cv::Point &pt) {
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

nav_msgs::msg::Path
MedialAxis::convertMapPathToRealPath(const std::vector<cv::Point> &path_map,
                                     const std_msgs::msg::Header &header) {

  nav_msgs::msg::Path path_world;
  path_world.header = header;

  const size_t N = path_map.size();
  path_world.poses.resize(N);

  // Iterating up to N-1, so that the last entry exists
  for (size_t i = 0; i < N - 1; ++i) {
    const cv::Point &pt = path_map.at(i), &pt_next = path_map.at(i + 1);
    auto &pose = path_world.poses.at(i);
    pose.header = header;

    double x, y;
    costmap_->mapToWorld(pt.x, pt.y, x, y);
    pose.pose.position.x = x;
    pose.pose.position.y = y;

    // Assign yaw to point to the next entry
    double yaw = std::atan2(pt_next.y - pt.y, pt_next.x - pt.x);
    pose.pose.orientation.x = 0.0;
    pose.pose.orientation.y = 0.0;
    pose.pose.orientation.z = std::sin(yaw / 2.);
    pose.pose.orientation.w = std::cos(yaw / 2.);
  }

  return path_world;
}

} // namespace medial_axis_planner

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(medial_axis_planner::MedialAxis,
                       nav2_core::GlobalPlanner)
