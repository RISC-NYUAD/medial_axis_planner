cmake_minimum_required(VERSION 2.8.3)
project(pathfinder)
set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_FLAGS "-std=c++0x -Wno-psabi ${CMAKE_CXX_FLAGS}")
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
    sensor_msgs
    std_msgs
    nav_msgs
    roscpp
    tf
    octomap_server
    image_transport
    cv_bridge
)

find_package(Boost REQUIRED COMPONENTS thread)
find_package(OpenCV REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES flight_pkg
   CATKIN_DEPENDS
   roscpp
   sensor_msgs
   std_msgs
   roscpp
   tf
   octomap_server
   image_transport
   cv_bridge
)


include_directories(
 include src ${catkin_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} /usr/include/eigen3
)

add_executable(pathfinding_node src/pathfinding_node.cpp)
target_link_libraries(pathfinding_node ${catkin_LIBRARIES})
