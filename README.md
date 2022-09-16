# Path-Finding-2D-Slice-on-Voxel-Map
Path Finding on 2D Slice from Voxel Map

This node receives the two voxel based topics

	ros::Subscriber sub = nh.subscribe("/occupied_cells_vis_array", 1000, chatterCallback);
	ros::Subscriber subA = nh.subscribe("/free_cells_vis_array", 1000, chatterCallbackA);//needs mapping_node/publish_free_space: true in params.yaml

It then plots circles in the place where points exist in space between two height values

	float XA = msg->markers[i].points[j].y;
	float XB = msg->markers[i].points[j].x;
	float XC = msg->markers[i].points[j].z;
	XA = XA + 20; XA = multiplier * XA;     ///shift points to align well in image
	XB = XB + 10; XB = multiplier * XB;
	if(XC > 0.5 && XC < 0.9 ){ //height check
        //plot circle in boundary and free space points
        circle(plotBorder, Point(XA, XB), 10,  CV_RGB(142, 0, 0), -1);

 Then checks for pixels where are not blocked and fills in the openSpaceSpots list with free points where the UAV
 can pass through to go to the target

 Those points are then inserted in the path finding method which finds and plots the possible paths, given a downscaled
 version (BWA) of the original map (plotBorder), thresholded to black and white, where white the free space.
 Source is the target point where the UAV needs to go.

 	vector<SPoint> startPoints;
 	for (int i=0;i<openSpaceSpots.size();i=i+1){
      		//scale points to downscaled image
        	startPoints.push_back({openSpaceSpots[i].y/dividme, openSpaceSpots[i].x/dividme});
 	}
 	int dist = BFSMULTIPLE(BWA, source, startPoints, plotBorder, dividme);

  When all paths to the destination are found, the system plots them in the while (!q.empty()) loop
  and saves the paths in the savedPaths list.

  Also if no new points are found, the list is not updated and is just ploted in the following if section,
  from the previous saved paths list

  	if(foundPathCount == dest.size() && foundPathCount > 0){

  NOTE: Only the .cpp file is required, the CMakeLists.txt is to showcase the addition of the node to the build for katkin.
  The.cpp must be copied to the "octomap_wrapper" in "src" folder before building with katkin.

  To run the node, paste the following command in a terminal after katkin build and having the Gazobo and Octomap running

  	rosrun octomap_wrapper mapping_nodeA

GAZEBO and OCTOMAP reference

	to build:
	cd catkin_ws ; catkin build

	to run:
	roslaunch uav_simulator mapping_auto.launch
	-- this starts the autonomous movement simulation. Need to connect the joystick and click the button on the left side to start
	roslaunch uav_simulator velocity_flight.launch
	-- this starts a simulation and the UAV can be unlocked with the same joystick button, then flown with the main joystick (harder to control)

	octomap:
	roslaunch octomap_wrapper mapping.launch

  TO DO:
  - Finalize the expansion of the path finding algorithm restreiction only on the convex hull of the needed regions, with an offset.
  - Add more to the way height threshold is chosen for the 2D slicing and possible ways to cover for the full 3D height range
  - Add the UAV position and calculate the shortest paths in the occupied area from the UAV to the discovered free space interface points
  - Calculate the final shortest path from the UAV to the target
  - Account for a non point target, e.g. a polyline boundary
