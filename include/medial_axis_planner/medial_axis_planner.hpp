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
  cv::Mat traversible_lut_, occupied_lut_;

  // Parameters
  double interpolation_resolution_;
};

} // namespace medial_axis_planner
