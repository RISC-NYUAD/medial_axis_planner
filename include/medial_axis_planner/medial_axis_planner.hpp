#pragma once

#include <memory>
#include <string>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

#include <nav2_core/global_planner.hpp>
#include <nav2_costmap_2d/costmap_2d_ros.hpp>
#include <nav2_util/lifecycle_node.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <opencv2/core/mat.hpp>

namespace medial_axis_planner {

// Region with external and internal contours
struct RegionContours {
  std::vector<cv::Point> external; // Points at the outermost boundary
  std::vector<std::vector<cv::Point>>
      internal; // Points at the boundary of holes
  bool valid;
};

class MedialAxis : public nav2_core::GlobalPlanner {
public:
  MedialAxis();
  ~MedialAxis();

  void configure(
      const rclcpp_lifecycle::LifecycleNode::WeakPtr &parent, std::string name,
      std::shared_ptr<tf2_ros::Buffer> tf,
      std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  // plugin cleanup
  void cleanup() override;

  // plugin activate
  void activate() override;

  // This method creates path for given start and goal pose.
  nav_msgs::msg::Path
  createPlan(const geometry_msgs::msg::PoseStamped &start,
             const geometry_msgs::msg::PoseStamped &goal) override;

  // plugin deactivate
  void deactivate() override;

private:
  // Extract the contour containing the given coordinate
  RegionContours extractRegionContainingCoordinate(const cv::Mat &traversible,
                                                   const cv::Point &coordinate,
                                                   bool simplify);

  // Give coordinates of the closest line-of-sight point from the set
  cv::Point
  findClosesetLineofSightFrom(const std::vector<cv::Point> &candidates,
                              const cv::Point &target,
                              const cv::Mat &valid_region);

  // Checks if the linear trajectory between start and end points traverse
  // outside of the boundary defined by the image, and returns the straight
  // line if it is within the bounds
  std::vector<cv::Point> computeLineOfSightPath(const cv::Mat &region,
                                                const cv::Point &start,
                                                const cv::Point &end);

  // Dijkstra's algorithm to find a path along the medial axis
  std::vector<cv::Point>
  findPathAlongMedialAxis(const std::vector<cv::Point> &medial_axis_pts,
                          const cv::Point &start, const cv::Point &end);

  // Compute 8-neighbors of a given point
  std::vector<cv::Point> computeNeighbors(const cv::Point &pt);

  // Convert from map path to real (metric) path with yaw assignment
  nav_msgs::msg::Path
  convertMapPathToRealPath(const std::vector<cv::Point> &path_map,
                           const std_msgs::msg::Header &header);

  // Parent node pointer
  nav2_util::LifecycleNode::SharedPtr node_;

  // TF buffer
  std::shared_ptr<tf2_ros::Buffer> tf_;

  // Clock & logger
  rclcpp::Clock::SharedPtr clock_;
  rclcpp::Logger logger_{rclcpp::get_logger("MedialAxisPlanner")};

  // Introspective visual
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_publisher_;

  // Global costmap TODO: why not unique_ptr or shared_ptr?
  nav2_costmap_2d::Costmap2D *costmap_;

  // The global frame of the costmap
  std::string global_frame_, name_;

  // Lookup tables
  cv::Mat free_lut_, occupied_lut_, danger_lut_, unknown_lut_;

  // Parameters
  double interpolation_resolution_;
};

} // namespace medial_axis_planner
