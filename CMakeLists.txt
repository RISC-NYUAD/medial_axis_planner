cmake_minimum_required(VERSION 3.5)
project(medial_axis_planner)

# Export compile options for lsp
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(nav2_util REQUIRED)
find_package(pluginlib REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)

find_package(OpenCV REQUIRED)

set(DEPENDENCIES
  rclcpp
  rclcpp_action
  rclcpp_lifecycle
  tf2_ros
  nav_msgs
  nav2_msgs
  nav2_core
  nav2_costmap_2d
  nav2_util
  pluginlib
  OpenCV
  sensor_msgs
  image_transport
  cv_bridge
)

include_directories(
  include
)

# Targets
set(LIBRARY_NAME ${PROJECT_NAME}_plugin)

add_library(${LIBRARY_NAME} SHARED
  src/medial_axis_planner.cpp
)

ament_target_dependencies(
  ${LIBRARY_NAME}
  ${DEPENDENCIES}
)

target_compile_definitions(${LIBRARY_NAME} PUBLIC "PLUGINLIB__DISABLE_BOOST_FUNCTIONS")

pluginlib_export_plugin_description_file(nav2_core global_planner_plugin.xml)

install(
  TARGETS ${LIBRARY_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include/
)

install(
  FILES global_planner_plugin.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_export_include_directories(include)
ament_export_libraries(${LIBRARY_NAME})
ament_export_dependencies(${DEPENDENCIES})
ament_package()

# ament_auto_find_build_dependencies()

# # Targets
# ament_auto_add_library(${LIBRARY_NAME} SHARED src/medial_axis_planner.cpp)
# target_compile_definitions(${LIBRARY_NAME} PUBLIC "PLUGINLIB__DISABLE_BOOST_FUNCTIONS")

# pluginlib_export_plugin_description_file(nav2_core global_planner_plugin.xml)

# ament_auto_package()
