#include "medial_axis_planner/medial_axis_planner.hpp"
#include "nav2_util/node_utils.hpp"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

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
  // Traversible consists of free + unknown
  traversible_lut_ = cv::Mat(1, 256, CV_8U, cv::Scalar(0));
  traversible_lut_.at<uint8_t>(0, nav2_costmap_2d::FREE_SPACE) = 255;
  traversible_lut_.at<uint8_t>(0, nav2_costmap_2d::NO_INFORMATION) = 255;

  // Occupied consists of lethal
  occupied_lut_ = cv::Mat(1, 256, CV_8U, cv::Scalar(0));
  occupied_lut_.at<uint8_t>(0, nav2_costmap_2d::LETHAL_OBSTACLE) = 255;
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

  size_t size_x = costmap_->getSizeInCellsX(),
         size_y = costmap_->getSizeInCellsY();

  // For OpenCV mat converted in below method, x right, y down (z inwards)
  // Use cv::flip(src, src, 0) for z upwards
  cv::Mat costmap_raw(size_y, size_x, CV_8UC1, costmap_->getCharMap());

  // Lookup tables to extract individual layers
  cv::Mat occupied, traversible;
  cv::LUT(costmap_raw, occupied_lut_, occupied);
  cv::LUT(costmap_raw, traversible_lut_, traversible);

  // Visualize with cv2
  cv::Mat zeros(size_y, size_x, CV_8UC1, cv::Scalar(0));
  std::vector<cv::Mat> channels{zeros, traversible, occupied};
  cv::Mat visual;
  cv::merge(channels, visual);
  cv::flip(visual, visual, 0);

  std_msgs::msg::Header header;
  sensor_msgs::msg::Image::SharedPtr visual_msg =
      cv_bridge::CvImage(header, "bgr8", visual).toImageMsg();
  debug_publisher_->publish(*visual_msg.get());

  nav_msgs::msg::Path path;

  return path;
}

} // namespace medial_axis_planner

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(medial_axis_planner::MedialAxis,
                       nav2_core::GlobalPlanner)
