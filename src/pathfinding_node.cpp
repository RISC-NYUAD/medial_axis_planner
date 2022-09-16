/**
 * octomap_server: A Tool to serve 3D OctoMaps in ROS (binary and as
 * visualization) (inspired by the ROS map_saver)
 * @author A. Hornung, University of Freiburg, Copyright (C) 2009 - 2012.
 * @see http://octomap.sourceforge.net/
 * License: BSD
 */

/*
 * Copyright (c) 2009-2012, A. Hornung, University of Freiburg
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University of Freiburg nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <octomap_server/OctomapServer.h>
#include <ros/ros.h>
#include <std_msgs/Int16.h>

#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

#include "opencv2/features2d.hpp"
#include "ros/duration.h"
#include "ros/node_handle.h"
#include "std_msgs/String.h"
#include "visualization_msgs/MarkerArray.h"

int boundBoxMinX = 100000;
int boundBoxMinY = 100000;
int boundBoxMaxX = -100000;
int boundBoxMaxY = -100000;

////////////////// SEARCH PATH
// C++ program to find the shortest path between
// a given source cell to a destination cell.

#define ROW 60  // 45 with 16, 60 with 12
#define COL 106 // 80 with 16, 106 with 12

// To store matrix cell coordinates
struct SPoint {
  int x;
  int y;
};

// A Data Structure for queue used in BFS
struct queueNode {
  SPoint pt; // The coordinates of a cell
  int dist;  // cell's distance of from the source
  SPoint parentCoords;
  std::vector<cv::Point> pathPoints;
};

// check whether given cell (row, col) is a valid
// cell or not.
bool isValid(int row, int col) {
  // return true if row number and column number
  // is in range
  return (row >= 0) && (row < ROW) && (col >= 0) && (col < COL);
}

// These arrays are used to get row and column
// numbers of 4 neighbours of a given cell
int rowNum[] = {-1, 0, 0, 1, 1, -1, 1, -1};
int colNum[] = {0, -1, 1, 0, 1, -1, -1, 1};

int foundPathCount = 0;
std::vector<std::vector<cv::Point>> savedPaths;
int BFSMULTIPLE(cv::Mat mat, SPoint src, std::vector<SPoint> dest,
                cv::Mat plotBorder, int divider) {
  // check source and destination cell of the matrix have value 1
  int testerA = mat.at<uchar>(src.x, src.y);
  int testerB = mat.at<uchar>(dest[0].x, dest[0].y);
  if (!(testerA > 0) || !(testerB > 0))
    return -2;

  bool visited[ROW][COL];
  memset(visited, false, sizeof visited);

  // Mark the source cell as visited
  visited[src.x][src.y] = true;

  // Create a queue for BFS
  std::queue<queueNode> q;
  std::queue<queueNode> qA;

  // Distance of source cell is 0
  queueNode s = {src, 0};
  q.push(s); // Enqueue source cell

  // Do a BFS starting from source cell
  int countSaved = 0;
  if (foundPathCount == dest.size() && foundPathCount > 0) {
  } else {
    savedPaths.clear(); // clear saved if found new
  }
  int toggle = 0;
  while (!q.empty()) {
    queueNode curr = q.front();
    SPoint pt = curr.pt;

    cv::Point last_Point = cv::Point(curr.pt.x, curr.pt.y);
    float lastParentDist = curr.dist;

    // If we have reached the destination cell, we are done

    if (foundPathCount == dest.size() && foundPathCount > 0) {
      // foundPathCount = dest.size();

      // plot history
      std::cout << "savedPaths.size() = "
                << "SAME " << savedPaths.size() << std::endl;
      for (int i = 0; i < savedPaths.size(); i++) {
        std::vector<cv::Point> currPREV = savedPaths[i];
        std::cout << "currPREV.size() = "
                  << "SAME " << currPREV.size() << std::endl;
        for (int j = 0; j < currPREV.size(); j++) {
          circle(plotBorder, currPREV[j] * divider, 3, CV_RGB(110, 15, 225),
                 -1);

          // line
          int red = 0;
          if (i > savedPaths.size() / 2) {
            red = 255;
          }
          if (j > 0) {
            line(plotBorder, currPREV[j] * divider, currPREV[j - 1] * divider,
                 cv::Scalar(i / 1.5, 110, red), 2);
          }
        }
      }

      std::cout << "dest.size() = "
                << "SAME " << dest.size() << std::endl;
      return curr.dist;
    } else {
      // savedPaths.clear();//clear saved if found new
      // foundPathCount = dest.size();
      // cout << "dest.size() = " << dest.size() << endl;
    }
    int foundPoints = 0;
    for (int i = 0; i < dest.size(); i++) {
      if (pt.x == dest[i].x && pt.y == dest[i].y) {
        for (int j = 0; j < curr.pathPoints.size(); j++) {
          circle(plotBorder, curr.pathPoints[j] * divider, 3,
                 CV_RGB(110, 15, 225), -1);
          // line
          int red = 0;
          if (i > dest.size() / 2) {
            red = 255;
          }
          if (j > 0) {
            line(plotBorder, curr.pathPoints[j] * divider,
                 curr.pathPoints[j - 1] * divider,
                 cv::Scalar(i / 1.5, 110, red), 2);
          }
        }
        savedPaths.push_back(curr.pathPoints);
        circle(plotBorder, cv::Point(pt.y, pt.x) * divider, 9,
               CV_RGB(0, 255, 155),
               -1); // OPEN SPACE
        circle(plotBorder, cv::Point(pt.y, pt.x) * divider, 9,
               CV_RGB(11, 15, 155),
               1); // OPEN SPACE
        // waitKey(10);
        // return curr.dist;
        foundPoints++;
      }
    }
    // cout << "dest.size() = " << dest.size() << " foundPoints = " <<
    // foundPoints << " savedPaths size = " << savedPaths.size() << endl;
    if (dest.size() == savedPaths.size() &&
        savedPaths.size() > 0) { // dest.size()){
      foundPathCount = dest.size();
      return curr.dist;
    }
    q.pop();

    int countrToggle = 8;
    if (curr.dist % 2 != 0) { // toggle == 0){
      toggle = 1;
      countrToggle = 8;
    } else {      // if(toggle == 1){
      toggle = 0; // 2;
      countrToggle = 8;
    }
    // else{
    //	toggle=0;
    //	countrToggle = 3;
    //}

    for (int i = 0; i < countrToggle; i++) {
      int row = pt.x + rowNum[i];
      int col = pt.y + colNum[i];

      // if adjacent cell is valid, has path and not visited yet, enqueue it.
      bool isOne = false;
      int tester = mat.at<uchar>(row, col);
      if (tester > 254) {
        isOne = true;
      }
      if (isValid(row, col) && tester && !visited[row][col]) {
        curr.pathPoints.push_back(cv::Point(curr.pt.y, curr.pt.x));

        // mark cell as visited and enqueue it
        visited[row][col] = true;
        queueNode Adjcell = {
            {row, col}, curr.dist + 1, {pt.x, pt.y}, curr.pathPoints};
        q.push(Adjcell);
        qA.push(Adjcell);

        // TEXT
        cv::putText(
            plotBorder,                                // target image
            std::to_string(curr.dist),                 // text
            cv::Point(curr.pt.y, curr.pt.x) * divider, // top-left position
            cv::FONT_HERSHEY_DUPLEX, 0.23, CV_RGB(118, 185, 0), // font color
            1);
      }
    }
  }

  // cv::imshow("Orig", plotBorder);
  cv::waitKey(10);

  // Return -1 if destination cannot be reached
  return -1;
}

////////////////// END SEARCH PATH

cv::Mat plotBorder;
int xres = 1280;
int yres = 720;
float multiplier = 30;

void chatterCallback(
    const visualization_msgs::MarkerArray::ConstPtr
        &msg) // void chatterCallback(const std_msgs::String::ConstPtr& msg)
{
  cv::namedWindow("Orig", cv::WINDOW_NORMAL);

  plotBorder = cv::Scalar(255, 255, 255);

  for (int i = 0; i < msg->markers.size(); i++) {
    for (int j = 0; j < msg->markers[i].points.size(); j++) {
      // circle(plotBorder, Point(100,100), 5,  CV_RGB(142/1, 2/112, 220/121),
      // -1);
      float XA = msg->markers[i].points[j].y;
      float XB = msg->markers[i].points[j].x;
      float XC = msg->markers[i].points[j].z;
      XA = XA + 20;
      XA = multiplier * XA;
      XB = XB + 10;
      XB = multiplier * XB;
      // std::cout << "Point: " << XA << "," << XB << std::endl;
      if (XC > 0.5 && XC < 0.9) {
        circle(plotBorder, cv::Point(XA, XB), 10, CV_RGB(142, 0, 0), -1); // 11

        if (XA > boundBoxMaxX) {
          boundBoxMaxX = XA;
        }
        if (XA < boundBoxMinX) {
          boundBoxMinX = XA;
        }
        if (XB > boundBoxMaxY) {
          boundBoxMaxY = XB;
        }
        if (XB < boundBoxMinY) {
          boundBoxMinY = XB;
        }
      }
    }
  }
}

std::vector<cv::Point> openSpaceSpots;

void chatterCallbackA(
    const visualization_msgs::MarkerArray::ConstPtr
        &msg) // void chatterCallback(const std_msgs::String::ConstPtr& msg)
{
  for (int i = 0; i < msg->markers.size(); i++) {
    for (int j = 0; j < msg->markers[i].points.size(); j++) {
      float XA = msg->markers[i].points[j].y;
      float XB = msg->markers[i].points[j].x;
      float XC = msg->markers[i].points[j].z;
      XA = XA + 20;
      XA = multiplier * XA;
      XB = XB + 10;
      XB = multiplier * XB;
      if (XC > 0.5 && XC < 0.9) {
        circle(plotBorder, cv::Point(XA, XB), 9, CV_RGB(0, 255, 0),
               -1); // free space
      }
    }
  }

  // clear the spots found in the previous frame
  openSpaceSpots.clear();

  for (int i = 0; i < msg->markers.size(); i++) {
    for (int j = 0; j < msg->markers[i].points.size(); j++) {
      // circle(plotBorder, cv::Point(100,100), 5,  CV_RGB(142/1, 2/112,
      // 220/121), -1);
      float XA = msg->markers[i].points[j].y;
      float XB = msg->markers[i].points[j].x;
      float XC = msg->markers[i].points[j].z;
      XA = XA + 20;
      XA = multiplier * XA;
      XB = XB + 10;
      XB = multiplier * XB;
      // std::cout << "cv::Point: " << XA << "," << XB << std::endl;
      if (XC > 0.5 && XC < 0.9) {
        int foundBorderPoints = 0;
        int foundBorderPointsA = 0;
        for (int i1 = XB - 10; i1 < XB + 10; i1++) {
          for (int i2 = XA - 10; i2 < XA + 10; i2++) {
            if (i1 > 0 && i1 < yres && i2 > 0 && i2 < xres) {
              cv::Vec3b &color = plotBorder.at<cv::Vec3b>(i1, i2);
              if ((color[2] < 250 && color[2] > 10)) {
                foundBorderPoints++;
              }
              if (color[0] == 255 && color[1] == 255 && color[2] == 255) {
                foundBorderPointsA++;
              }
            }
          }
        }
        if (foundBorderPoints > 0) {
          // circle(plotBorder, Point(XA, XB), 4,  CV_RGB(0, 0, 255), -1);
        } else {
          if (foundBorderPointsA > 0) {
            circle(plotBorder, cv::Point(XA, XB), 9, CV_RGB(0, 255, 255),
                   -1); // OPEN SPACE ------- SHOULD BE CYAN --- BUT need to be
                        // free to start the path finder
            openSpaceSpots.push_back(cv::Point(XA, XB));
          } else {
            // circle(plotBorder, Point(XA, XB), 8,  CV_RGB(0, 255, 0), -1);
            // //free space
          }
        }
      }
    }
  }

  // PLOT target
  // circle(plotBorder, Point(100, 400), 60,  CV_RGB(0, 155, 255), -1);//OPEN
  // SPACE

  // Mat BW;
  // resize(plotBorder, BW,Size(1280/4, 720/4));
  // cvtColor(BW, BW,cv::COLOR_RGB2GRAY);
  // threshold(BW,BW,254, 255, THRESH_BINARY);
  // cv::imshow("OrigA1", BW);
  // cv::ximgproc::thinning(BW,BW,cv::ximgproc::THINNING_GUOHALL);
  // //cv::ximgproc::THINNING_ZHANGSUEN, THINNING_GUOHALL bitwise_not(BW,BW);
  // resize(BW, BW,Size(1280, 720));
  // cv::bitwise_and(plotBorder, plotBorder, plotBorder, BW);
  // plotBorder.copyTo(plotBorder,BW);

  // PATH
  int dividme = 12; // 16;
  SPoint source = {16, 6};
  SPoint dest = {36, 78}; // SPoint dest = {43,78};

  // BOUNDING BOX
  if (source.y * dividme > boundBoxMaxX) {
    boundBoxMaxX = source.y * dividme;
  }
  if (source.y * dividme < boundBoxMinX) {
    boundBoxMinX = source.y * dividme;
  }
  if (source.x * dividme > boundBoxMaxY) {
    boundBoxMaxY = source.x * dividme;
  }
  if (source.x * dividme < boundBoxMinY) {
    boundBoxMinY = source.x * dividme;
  }
  // expand based on downscaling
  float DownscaleOffset =
      15 * 3; // downscale pixels multipled by max used cirlces
  int boundBoxMinXA = boundBoxMinX - DownscaleOffset;
  int boundBoxMinYA = boundBoxMinY - DownscaleOffset;
  int boundBoxMaxXA = boundBoxMaxX + DownscaleOffset;
  int boundBoxMaxYA = boundBoxMaxY + DownscaleOffset;
  std::cout << "bound Box = " << boundBoxMinXA << "," << boundBoxMinYA << ","
            << boundBoxMaxXA << "," << boundBoxMaxYA << std::endl;

  // Iterate paths for open area
  ////// PATH

  cv::Mat BWA;

  // RESET CYAN to WHITE to enable free space in interface
  for (int i = 0; i < openSpaceSpots.size(); i++) {
    circle(plotBorder, openSpaceSpots[i], 9, CV_RGB(255, 255, 255), -1);
  }

  // Mat cropped = plotBorder(Rect(boundBoxMinXA, boundBoxMinYA, boundBoxMaxXA,
  // boundBoxMaxYA));
  cv::Mat cropped = plotBorder(
      cv::Rect(boundBoxMinXA, boundBoxMinYA, boundBoxMaxXA, boundBoxMaxYA));
  cv::imshow("cropped", cropped);

  resize(plotBorder, BWA, cv::Size(1280 / dividme, 720 / dividme));
  cvtColor(BWA, BWA, cv::COLOR_RGB2GRAY);
  threshold(BWA, BWA, 254, 255, cv::THRESH_BINARY);
  // bitwise_not(BWA,BWA);
  // int tester = BWA.at<uchar>(2, 2);
  // int tester1 = BWA.at<uchar>(22, 22);
  // int dist = BFS(BWA, source, dest, plotBorder, dividme);
  // std::cout << "Shortest Path is " << tester << " , " << tester  << " dist "
  // << dist <<endl;//dist ;
  std::cout << "openSpaceSpots.size() = " << openSpaceSpots.size() << std::endl;
  std::vector<SPoint> startPoints;
  for (int i = 0; i < openSpaceSpots.size();
       i = i + 1) { // (openSpaceSpots.size() / 10 ) ){
    startPoints.push_back(
        {openSpaceSpots[i].y / dividme, openSpaceSpots[i].x / dividme});
  }
  int dist = BFSMULTIPLE(BWA, source, startPoints, plotBorder, dividme);

  // PLOT BOUND BOX
  cv::line(
      plotBorder,
      cv::Point(boundBoxMinXA, boundBoxMinYA), // curr.pathPoints[j] * divider,
      cv::Point(boundBoxMinXA, boundBoxMaxYA), cv::Scalar(1.5, 110, 1), 2);
  cv::line(
      plotBorder,
      cv::Point(boundBoxMinXA, boundBoxMaxYA), // curr.pathPoints[j] * divider,
      cv::Point(boundBoxMaxXA, boundBoxMaxYA), cv::Scalar(1.5, 110, 1), 2);
  cv::line(
      plotBorder,
      cv::Point(boundBoxMaxXA, boundBoxMaxYA), // curr.pathPoints[j] * divider,
      cv::Point(boundBoxMaxXA, boundBoxMinYA), cv::Scalar(1.5, 110, 1), 2);
  cv::line(
      plotBorder,
      cv::Point(boundBoxMaxXA, boundBoxMinYA), // curr.pathPoints[j] * divider,
      cv::Point(boundBoxMinXA, boundBoxMinYA), cv::Scalar(1.5, 110, 1), 2);
  cv::circle(plotBorder, cv::Point(source.y * dividme, source.x * dividme), 9,
             CV_RGB(255, 1, 1), -1);

  // SPoint sourceA = {6, 6};
  // SPoint destA = {58,68};
  // int distA = BFS(BWA, sourceA, destA, plotBorder, dividme);

  cv::imshow("OrigA", BWA);
  ////// END PATH

  cv::imshow("Orig", plotBorder); // cv::imshow("Orig", plotBorder);
  // cv::imshow("OrigA", BW);
  cv::waitKey(10);
}

//////////// MAIN ///////////////
int main(int argc, char **argv) {
  ros::init(argc, argv, "pathfinding_node");
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  // v0.1
  cv::Mat plotBorderA(yres, xres, CV_8UC3, cv::Scalar(1, 1, 1));
  plotBorderA.copyTo(plotBorder);

  ROS_INFO("[Mapping] Initializing octomap");

  ros::Publisher chatter_pub = nh.advertise<std_msgs::String>("chatter", 1000);

  ros::Rate loop_rate(10);
  int count = 0;

  ros::Subscriber sub =
      nh.subscribe("/occupied_cells_vis_array", 1000, chatterCallback);
  ros::Subscriber subA =
      nh.subscribe("/free_cells_vis_array", 1000, chatterCallbackA);

  while (ros::ok()) {
    std_msgs::String msg;
    std::stringstream ss;
    ss << "hello world " << count;
    msg.data = ss.str();
    chatter_pub.publish(msg);
    count++;

    ros::spinOnce();
    loop_rate.sleep();
  }
  return 0;
}
