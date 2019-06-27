#include <ros/ros.h>
#include <std_msgs/String.h>
#include <chrono>
#include <geometry_msgs/Twist.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>
#include <tf/transform_listener.h>
#include <interactive_cleanup/InteractiveCleanupMsg.h>
#include <nodelet/nodelet.h>



//add hagi
#include <sensor_msgs/CameraInfo.h>
#include <darknet_ros_msgs/BoundingBox.h>
#include <darknet_ros_msgs/BoundingBoxes.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
using namespace std::chrono;

class InteractiveCleanupSample
{
private:
  enum Step
  {
    Initialize,
    Ready,
    WaitForInstruction,
    GoToRoom1,
    GoToRoom2,
    MoveToInFrontOfTarget,
    Grasp,
    WaitForGrasping,
    ComeBack,
    ReleaseObject,
    WaitForReleasing,
    TaskFinished,
  };

  const std::string MSG_ARE_YOU_READY    = "Are_you_ready?";
  const std::string MSG_CLEAN_UP         = "Clean_up!";
  const std::string MSG_TASK_SUCCEEDED   = "Task_succeeded";
  const std::string MSG_TASK_FAILED      = "Task_failed";
  const std::string MSG_MISSION_COMPLETE = "Mission_complete";

  const std::string MSG_I_AM_READY      = "I_am_ready";
  const std::string MSG_OBJECT_GRASPED  = "Object_grasped";
  const std::string MSG_TASK_FINISHED   = "Task_finished";
 
  //add
  const int OBJECTS_INFO_UPDATING_INTERVAL = 500; //[ms]
  const int MAX_OBJECTS_NUM = 10;
  const double PROBABILITY_THRESHOLD = 0.3;
  const double ROTATION_VEL = 0.05;
  const double MOVE_VEL = 0.3;
  
  //
  trajectory_msgs::JointTrajectory arm_joint_trajectory_;
  trajectory_msgs::JointTrajectory gripper_joint_trajectory_;

  int step_;

  bool is_started_;
  bool has_been_instructed_;
  bool is_finished_;
  bool is_failed_;
  
  //add hagi
  int rgb_camera_height_, rgb_camera_width_;
  time_point<system_clock> latest_time_of_bounding_boxes_;
  darknet_ros_msgs::BoundingBoxes bounding_boxes_data_;
  time_point<system_clock> latest_time_of_point_cloud_;
  sensor_msgs::PointCloud2 point_cloud_data_;
  template <class T> static T clamp(const T val, const T min, const T max);
  //

  void init()
  {
    // Arm Joint Trajectory
    std::vector<std::string> arm_joint_names {"arm_lift_joint", "arm_flex_joint", "arm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"};

    trajectory_msgs::JointTrajectoryPoint arm_joint_point;

    arm_joint_trajectory_.joint_names = arm_joint_names;
    arm_joint_trajectory_.points.push_back(arm_joint_point);

    // Gripper Joint Trajectory
    std::vector<std::string> gripper_joint_names {"hand_l_proximal_joint", "hand_r_proximal_joint"};

    trajectory_msgs::JointTrajectoryPoint gripper_joint_point;

    gripper_joint_trajectory_.joint_names = gripper_joint_names;
    gripper_joint_trajectory_.points.push_back(gripper_joint_point);

    step_ = Initialize;

    reset();
  }

  void reset()
  {
    is_started_          = false;
    has_been_instructed_ = false;
    is_finished_         = false;
    is_failed_           = false;

    std::vector<double> arm_positions { 0.0, 0.0, 0.0, 0.0, 0.0 };
    arm_joint_trajectory_.points[0].positions = arm_positions;

    std::vector<double> gripper_positions { 0.0, 0.0 };
    gripper_joint_trajectory_.points[0].positions = gripper_positions;
  }


  void messageCallback(const interactive_cleanup::InteractiveCleanupMsg::ConstPtr& message)
  {
    ROS_INFO("Subscribe message:%s, %s", message->message.c_str(), message->detail.c_str());

    if(message->message.c_str()==MSG_ARE_YOU_READY)
    {
      if(step_==Ready)
      {
        is_started_ = true;
      }
    }
    if(message->message.c_str()==MSG_CLEAN_UP)
    {
      if(step_==WaitForInstruction)
      {
        has_been_instructed_ = true;
      }
    }
    if(message->message.c_str()==MSG_TASK_SUCCEEDED)
    {
      if(step_==TaskFinished)
      {
        is_finished_ = true;
      }
    }
    if(message->message.c_str()==MSG_TASK_FAILED)
    {
      is_failed_ = true;
    }
    if(message->message.c_str()==MSG_MISSION_COMPLETE)
    {
      exit(EXIT_SUCCESS);
    }
  }

  void sendMessage(ros::Publisher &publisher, const std::string &message)
  {
    ROS_INFO("Send message:%s", message.c_str());

    interactive_cleanup::InteractiveCleanupMsg interactive_cleanup_msg;
    interactive_cleanup_msg.message = message;
    publisher.publish(interactive_cleanup_msg);
  }

  tf::StampedTransform getTfBase(tf::TransformListener &tf_listener)
  {
    tf::StampedTransform tf_transform;

    try
    {
      tf_listener.lookupTransform("/odom", "/base_footprint", ros::Time(0), tf_transform);
    }
    catch (tf::TransformException &ex)
    {
      ROS_ERROR("%s",ex.what());
    }

    return tf_transform;
  }

  void moveBase(ros::Publisher &publisher, double linear_x, double linear_y, double linear_z, double angular_x, double angular_y, double angular_z)
  {
    geometry_msgs::Twist twist;

    twist.linear.x  = linear_x;
    twist.linear.y  = linear_y;
    twist.linear.z  = linear_z;
    twist.angular.x = angular_x;
    twist.angular.y = angular_y;
    twist.angular.z = angular_z;

    publisher.publish(twist);
  }
void movegoal(ros::Publisher &publisher, double px, double py,double pz,double ow)
  {
    geometry_msgs::PoseStamped goal_point;//位置情報保存用変数
    //ros::Duration duration;
    //duration.sec = 5;
    goal_point.pose.position.x = px;//目標地点
    goal_point.pose.position.y = py;//目標地点
    goal_point.pose.position.z = pz;//
    goal_point.pose.orientation.w = ow;
    goal_point.header.frame_id = "map";
    publisher.publish(goal_point);///"move_base_simple/goal"
    ROS_INFO("move to goal");
    
  }
  void stopBase(ros::Publisher &publisher)
  {
    moveBase(publisher, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  }

  void moveArm(ros::Publisher &publisher, const std::vector<double> &positions, ros::Duration &duration)
  {
    arm_joint_trajectory_.points[0].positions = positions;
    arm_joint_trajectory_.points[0].time_from_start = duration;

    publisher.publish(arm_joint_trajectory_);
  }

  void moveArm(ros::Publisher &publisher, const std::vector<double> &positions)
  {
    ros::Duration duration;
    duration.sec = 1;
    moveArm(publisher, positions, duration);
  }

  void grasp(ros::Publisher &publisher)
  {
    ros::Duration duration;
    duration.sec = 2;
    std::vector<double> gripper_positions { -0.05, +0.05 };
    gripper_joint_trajectory_.points[0].positions = gripper_positions;
    gripper_joint_trajectory_.points[0].time_from_start = duration;

    publisher.publish(gripper_joint_trajectory_);
  }

  void openHand(ros::Publisher &publisher)
  {
    ros::Duration duration;
    duration.sec = 2;
    std::vector<double> gripper_positions { +0.611, -0.611 };
    gripper_joint_trajectory_.points[0].positions = gripper_positions;
    gripper_joint_trajectory_.points[0].time_from_start = duration;

    publisher.publish(gripper_joint_trajectory_);
  }
  
  //add hagi
  void rgbCameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& camera_info)
  {
	  rgb_camera_height_ = camera_info->height;
	  rgb_camera_width_  = camera_info->width;
  }
  
   
  void boundingBoxesCallback(const darknet_ros_msgs::BoundingBoxes::ConstPtr& bounding_boxes)
  {
	  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - latest_time_of_bounding_boxes_).count();
	  
	  if(duration < OBJECTS_INFO_UPDATING_INTERVAL){ return; }

	  latest_time_of_bounding_boxes_ = system_clock::now();
	  
	  //ROS_INFO("boundingBoxesCallback size=%d", (int)bounding_boxes->boundingBoxes.size());

	  int data_count = std::min<int>(bounding_boxes->boundingBoxes.size(), MAX_OBJECTS_NUM);

	  bounding_boxes_data_.boundingBoxes.resize(data_count);
	  
	  std::string detected_objects_ = "";

	  for(int i=0; i<data_count; i++)
	  {
		bounding_boxes_data_.boundingBoxes[i].Class       = bounding_boxes->boundingBoxes[i].Class;
		bounding_boxes_data_.boundingBoxes[i].probability = bounding_boxes->boundingBoxes[i].probability;
		bounding_boxes_data_.boundingBoxes[i].xmin        = bounding_boxes->boundingBoxes[i].xmin;
		bounding_boxes_data_.boundingBoxes[i].ymin        = bounding_boxes->boundingBoxes[i].ymin;
		bounding_boxes_data_.boundingBoxes[i].xmax        = bounding_boxes->boundingBoxes[i].xmax;
		bounding_boxes_data_.boundingBoxes[i].ymax        = bounding_boxes->boundingBoxes[i].ymax;
		puts(("class=" + bounding_boxes_data_.boundingBoxes[i].Class).c_str());
	  }
  }
  
void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& point_cloud)
{
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now() - latest_time_of_point_cloud_).count();

  if(duration < OBJECTS_INFO_UPDATING_INTERVAL){ return; }

  latest_time_of_point_cloud_ = system_clock::now();

  //ROS_INFO("pointCloudCallback row_step=%d", (int)point_cloud->row_step);

  point_cloud_data_.header.seq        = point_cloud->header.seq;
  point_cloud_data_.header.stamp.sec  = point_cloud->header.stamp.sec;
  point_cloud_data_.header.stamp.nsec = point_cloud->header.stamp.nsec;
  point_cloud_data_.header.frame_id   = point_cloud->header.frame_id;

  point_cloud_data_.height = point_cloud->height;
  point_cloud_data_.width  = point_cloud->width;

  point_cloud_data_.fields.resize(point_cloud->fields.size());

  for(int i=0; i<point_cloud->fields.size(); i++)
  {
    point_cloud_data_.fields[i].name     = point_cloud->fields[i].name;
    point_cloud_data_.fields[i].offset   = point_cloud->fields[i].offset;
    point_cloud_data_.fields[i].datatype = point_cloud->fields[i].datatype;
    point_cloud_data_.fields[i].count    = point_cloud->fields[i].count;
  }

  point_cloud_data_.is_bigendian = point_cloud->is_bigendian;
  point_cloud_data_.point_step   = point_cloud->point_step;
  point_cloud_data_.row_step     = point_cloud->row_step;

  int data_count = point_cloud->row_step * point_cloud->height;

  point_cloud_data_.data.resize(data_count);
  std::memcpy(&point_cloud_data_.data[0], &point_cloud->data[0], data_count * sizeof(uint8_t));

  point_cloud_data_.is_dense = point_cloud->is_dense;
}

  float findPerson()
  {	  
	int xcenter, ycenter, shita;
	float rad;

	for(int i=0; i<bounding_boxes_data_.boundingBoxes.size(); i++)
	{
		if(strcmp((bounding_boxes_data_.boundingBoxes[i].Class).c_str(),"person")==0){
			puts(("class=" + bounding_boxes_data_.boundingBoxes[i].Class).c_str());
			xcenter = (bounding_boxes_data_.boundingBoxes[i].xmin+bounding_boxes_data_.boundingBoxes[i].xmax)/2;
			ycenter = (bounding_boxes_data_.boundingBoxes[i].ymin+bounding_boxes_data_.boundingBoxes[i].ymax)/2;		  
			shita = (xcenter-rgb_camera_width_/2)*0.091;
			rad = shita*M_PI/180;
			//ROS_INFO("xcenter=%d, ycenter=%d, shita=%d, rad=%f", xcenter, ycenter, shita, rad);
		}
	}
	return rad;
  }

bool get3dPositionFromScreenPosition(geometry_msgs::Vector3 &position3d, const sensor_msgs::PointCloud2 point_cloud, const int x, const int y)
{
  if(point_cloud.header.seq==0)
  {
    puts("No point cloud data.");
    return false;
  }

  // -- PointCloud2 memo --
  // height: 360, width: 480
  // point_step=16, row_step=7680(=16*480)
  int point_data_start_position = y * point_cloud.row_step + x * point_cloud.point_step;

  int xpos = point_data_start_position + point_cloud.fields[0].offset;
  int ypos = point_data_start_position + point_cloud.fields[1].offset;
  int zpos = point_data_start_position + point_cloud.fields[2].offset;

  float pos3d_x, pos3d_y, pos3d_z;

  memcpy(&pos3d_x, &point_cloud.data[xpos], sizeof(float));
  memcpy(&pos3d_y, &point_cloud.data[ypos], sizeof(float));
  memcpy(&pos3d_z, &point_cloud.data[zpos], sizeof(float));

  position3d.x = (double)pos3d_x;
  position3d.y = (double)pos3d_y;
  position3d.z = (double)pos3d_z;

  if(std::isnan(position3d.x) || std::isnan(position3d.y) || std::isnan(position3d.z))
  {
//    puts("Point cloud data is nan.");
    return false;
  }

  return true;
}
  
bool findTarget(geometry_msgs::Vector3 &point_cloud_pos, const std::string &target_name)
{
  for(int i=0; i<bounding_boxes_data_.boundingBoxes.size(); i++)
  {
    if(bounding_boxes_data_.boundingBoxes[i].Class == target_name &&
       bounding_boxes_data_.boundingBoxes[i].probability > PROBABILITY_THRESHOLD)
    {
      puts(("Found the target. name= " + target_name).c_str());

      int center_x = (bounding_boxes_data_.boundingBoxes[i].xmax + bounding_boxes_data_.boundingBoxes[i].xmin) / 2;
      int center_y = (bounding_boxes_data_.boundingBoxes[i].ymax + bounding_boxes_data_.boundingBoxes[i].ymin) / 2;

//      puts(("x=" + std::to_string(center_x) + ", y=" + std::to_string(center_y)).c_str());

      int point_cloud_screen_x = clamp<int>((int)((float)center_x * point_cloud_data_.width  / rgb_camera_width_),  0, point_cloud_data_.width-1);
      int point_cloud_screen_y = clamp<int>((int)((float)center_y * point_cloud_data_.height / rgb_camera_height_), 0, point_cloud_data_.height-1);

      // the center
      bool is_succeeded = get3dPositionFromScreenPosition(point_cloud_pos, point_cloud_data_, point_cloud_screen_x, point_cloud_screen_y);
//      puts(("x=" + std::to_string(point_cloud_pos.x) + ", y=" + std::to_string(point_cloud_pos.y) + ", z=" + std::to_string(point_cloud_pos.z)).c_str());

      if(is_succeeded) { return true; }

      // Around the center (1/4)
      int play_x = (bounding_boxes_data_.boundingBoxes[i].xmax - center_x) / 4;
      int play_y = (bounding_boxes_data_.boundingBoxes[i].ymax - center_y) / 4;

      for(int yi=-play_y; yi<=+play_y; yi+=play_y)
      {
        for(int xi=-play_x; xi<=+play_x; xi+=play_x)
        {
          is_succeeded = get3dPositionFromScreenPosition(point_cloud_pos, point_cloud_data_, point_cloud_screen_x + xi, point_cloud_screen_y + yi);
          if(is_succeeded) { return true; }
        }
      }

      // Around the center (1/2)
      play_x = (bounding_boxes_data_.boundingBoxes[i].xmax - center_x) / 2;
      play_y = (bounding_boxes_data_.boundingBoxes[i].ymax - center_y) / 2;

      for(int yi=-play_y; yi<=+play_y; yi+=play_y)
      {
        for(int xi=-play_x; xi<=+play_x; xi+=play_x)
        {
          is_succeeded = get3dPositionFromScreenPosition(point_cloud_pos, point_cloud_data_, point_cloud_screen_x + xi, point_cloud_screen_y + yi);
          if(is_succeeded) { return true; }
        }
      }

      puts("Failed to get point cloud data.");
      return false;
    }
  }

  //puts(("Couldn't find " + target_name + ". Or low probability.").c_str());
  //puts(("objects=" + getDetectedObjectsList()).c_str());

  return false;
}

bool faceTowardObject(tf::StampedTransform &tf_transform, ros::Publisher &pub_base_twist, const std::string &target_name)
{
  geometry_msgs::Vector3 target_pos;
  if(!findTarget(target_pos, target_name))
  {
    puts(("Couldn't find target."));
    return false;
  }
  else{
	double rad;
	rad = atan2(target_pos.x, target_pos.z);
    ROS_INFO("target x=%lf, y=%lf, z=%lf, rad=%lf", target_pos.x, target_pos.y, target_pos.z, rad);
	double r,p,y;
	tf::Matrix3x3(tf_transform.getRotation()).getRPY(r, p, y);

	if(fabs(y) < fabs(rad)){
		if(rad > 0){
		  moveBase(pub_base_twist, 0.0, 0.0, 0.0, 0.0, 0.0, -ROTATION_VEL*0.1);
		}
		else{
		  moveBase(pub_base_twist, 0.0, 0.0, 0.0, 0.0, 0.0, ROTATION_VEL);
		}
		return false;
	}
	else{
		stopBase(pub_base_twist);
		return true;
	}
  }
}

bool moveTowardObject(tf::StampedTransform &tf_transform, ros::Publisher &pub_base_twist, const std::string &target_name)
{
  geometry_msgs::Vector3 target_pos;
  if(!findTarget(target_pos, target_name))
  {
    puts(("Couldn't find target."));
    return false;
  }
  else{
	double rad;
	rad = atan2(target_pos.x, target_pos.z);
    ROS_INFO("target x=%lf, y=%lf, z=%lf, rad=%lf", target_pos.x, target_pos.y, target_pos.z, rad);

	if(tf_transform.getOrigin().x() < (target_pos.z-0.3)){
		moveBase(pub_base_twist, MOVE_VEL, 0.0, 0.0, 0.0, 0.0, 0.0);
		return false;
	}
	else{
		stopBase(pub_base_twist);
		return true;
	}
  }
}

std::string getDetectedObjectsList()
{
  std::string detected_objects_ = "";

  for(int i=0; i<bounding_boxes_data_.boundingBoxes.size(); i++)
  {
    detected_objects_
      += bounding_boxes_data_.boundingBoxes[i].Class + ": "
      + std::to_string((int)std::floor(bounding_boxes_data_.boundingBoxes[i].probability * 100)) + "%   ";
  }
  return detected_objects_;
}

public:
  int run(int argc, char **argv)
  {
	
    ros::init(argc, argv, "interactive_cleanup_sample");

    ros::NodeHandle node_handle;

    ros::Rate loop_rate(10);

    std::string sub_msg_to_robot_topic_name;
    std::string pub_msg_to_moderator_topic_name;
    std::string pub_base_twist_topic_name;
    std::string pub_arm_trajectory_topic_name;
    std::string pub_gripper_trajectory_topic_name;

    //add hagi
    std::string sub_rgb_camera_info_topic_name;
    std::string sub_bounding_boxes_topic_name;
    std::string sub_point_cloud_topic_name;
    std::string pub_goal_point_topic_name;//宣言
    //
    

    node_handle.param<std::string>("sub_msg_to_robot_topic_name",       sub_msg_to_robot_topic_name,       "/interactive_cleanup/message/to_robot");
    node_handle.param<std::string>("pub_msg_to_moderator_topic_name",   pub_msg_to_moderator_topic_name,   "/interactive_cleanup/message/to_moderator");
    node_handle.param<std::string>("pub_base_twist_topic_name",         pub_base_twist_topic_name,         "/hsrb/opt_command_velocity");
    node_handle.param<std::string>("pub_arm_trajectory_topic_name",     pub_arm_trajectory_topic_name,     "/hsrb/arm_trajectory_controller/command");
    node_handle.param<std::string>("pub_gripper_trajectory_topic_name", pub_gripper_trajectory_topic_name, "/hsrb/gripper_trajectory_controller/command");
//add
    node_handle.param<std::string>("sub_rgb_camera_info_topic_name",    sub_rgb_camera_info_topic_name,    "/hsrb/head_rgbd_sensor/rgb/camera_info");
    node_handle.param<std::string>("sub_bounding_boxes_topic_name",     sub_bounding_boxes_topic_name,     "/darknet_ros/bounding_boxes");
    node_handle.param<std::string>("sub_point_cloud_topic_name",     sub_point_cloud_topic_name,     "/hsrb/head_rgbd_sensor/depth_registered/points");
    node_handle.param<std::string>("pub_goal_point_topic_name",      pub_goal_point_topic_name,     "/move_base_simple/goal");

    init();

    ros::Time waiting_start_time;

    ROS_INFO("Interactive Cleanup Sample start!");

    ros::Subscriber sub_msg                = node_handle.subscribe<interactive_cleanup::InteractiveCleanupMsg>(sub_msg_to_robot_topic_name, 100, &InteractiveCleanupSample::messageCallback, this);
    ros::Publisher  pub_msg                = node_handle.advertise<interactive_cleanup::InteractiveCleanupMsg>(pub_msg_to_moderator_topic_name, 10);
    ros::Publisher  pub_base_twist         = node_handle.advertise<geometry_msgs::Twist>            (pub_base_twist_topic_name, 10);
    ros::Publisher  pub_arm_trajectory     = node_handle.advertise<trajectory_msgs::JointTrajectory>(pub_arm_trajectory_topic_name, 10);
    ros::Publisher  pub_gripper_trajectory = node_handle.advertise<trajectory_msgs::JointTrajectory>(pub_gripper_trajectory_topic_name, 10);
    //add
    ros::Publisher  pub_goal_point          =node_handle.advertise<geometry_msgs::PoseStamped>(pub_goal_point_topic_name, 10);//位置情報保存用変数の宣言
    //add hagi
    ros::Subscriber sub_rgb_camera_info    = node_handle.subscribe(sub_rgb_camera_info_topic_name, 10, &InteractiveCleanupSample::rgbCameraInfoCallback, this);
    ros::Subscriber sub_bounding_boxes  = node_handle.subscribe(sub_bounding_boxes_topic_name,  10, &InteractiveCleanupSample::boundingBoxesCallback, this);
    ros::Subscriber sub_point_cloud  = node_handle.subscribe(sub_point_cloud_topic_name,  10, &InteractiveCleanupSample::pointCloudCallback, this);
	//


    tf::TransformListener tf_listener;

    while (ros::ok())
    {
      if(is_failed_)
      {
        ROS_INFO("Task failed!");
        step_ = Initialize;
      }

      switch(step_)
      {
        case Initialize:
        {
          reset();
          step_++;
          break;
        }
        case Ready:
        {
          if(is_started_)
          {
            sendMessage(pub_msg, MSG_I_AM_READY);

            ROS_INFO("Task start!");

            step_++;
          }
          break;
        }
        case WaitForInstruction:
        {
          ROS_INFO("WaitForInstruction");
		  // face toward person during instruction
		  tf::StampedTransform tf_transform = getTfBase(tf_listener);
		  /*add*/
		   movegoal(pub_goal_point, -0.9,1.8,0.0,-0.5916);
		   if(has_been_instructed_)step_++;
		   
	/*	   moveBase(pub_base_twist, +0.0, 0.0, 0.0, 0.0, 0.0, ROTATION_VEL);
		    if(faceTowardObject(tf_transform, pub_base_twist, "canned_juice")){
			ros::Duration(1.0).sleep();
			faceTowardObject(tf_transform, pub_base_twist, "canned_juice");
			movegoal(pub_goal_point,1.0,1.0);//movegoal(topic_name,num,num)
		      //if(faceTowardObject(tf_transform, pub_base_twist, "person")){
		       // if(moveTowardObject(tf_transform, pub_base_twist, "canned_juice"))//
			    //if(has_been_instructed_){
				//openHand(pub_gripper_trajectory);
				step_+=2;
			 }   */
		  //}
		   /*else if(faceTowardObject(tf_transform, pub_base_twist, "apple")){
		      //if(faceTowardObject(tf_transform, pub_base_twist, "person")){
		        if(moveTowardObject(tf_transform, pub_base_twist, "apple"))//
			    if(has_been_instructed_){
				openHand(pub_gripper_trajectory);
				step_+=2;
			}
		   
		  }
		   else if(faceTowardObject(tf_transform, pub_base_twist, "bear_doll")){
		      //if(faceTowardObject(tf_transform, pub_base_twist, "person")){
		        if(moveTowardObject(tf_transform, pub_base_twist, "bear_doll"))//
			    if(has_been_instructed_){
				openHand(pub_gripper_trajectory);
				step_+=2;
			}
		   
		  }
		  else if(faceTowardObject(tf_transform, pub_base_twist, "block_car")){
		      //if(faceTowardObject(tf_transform, pub_base_twist, "person")){
		        if(moveTowardObject(tf_transform, pub_base_twist, "block_car"))//
			    if(has_been_instructed_){
				openHand(pub_gripper_trajectory);
				step_+=2;
			}
		   
		  }
		  else if(faceTowardObject(tf_transform, pub_base_twist, "dog_doll")){
		      //if(faceTowardObject(tf_transform, pub_base_twist, "person")){
		        if(moveTowardObject(tf_transform, pub_base_twist, "dog_doll"))//
			    if(has_been_instructed_){
				openHand(pub_gripper_trajectory);
				step_+=2;
			}
		   
		  }
		  else if(faceTowardObject(tf_transform, pub_base_twist, "empty_mayonnaise")){
		      //if(faceTowardObject(tf_transform, pub_base_twist, "person")){
		        if(moveTowardObject(tf_transform, pub_base_twist, "empty_mayonnaise"))//
			    if(has_been_instructed_){
				openHand(pub_gripper_trajectory);
				step_+=2;
			}
		   
		  }*/
		 break;
        }
        case GoToRoom1:
        {
          ROS_INFO("GoToRoom1");
		  /*tf::StampedTransform tf_transform = getTfBase(tf_listener);
		  if(faceTowardObject(tf_transform, pub_base_twist, "person")){
			  if(moveTowardObject(tf_transform, pub_base_twist, "person")){
				  step_++;
			  }
		  }	*/
          break;
        }
       /* case GoToRoom2:
        {
          ROS_INFO("GoToRoom2");
          tf::StampedTransform tf_transform = getTfBase(tf_listener);

          if(tf_transform.getOrigin().y() >= -0.6)
          {
            moveBase(pub_base_twist, +1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
          }
          else
          {
            stopBase(pub_base_twist);

            step_++;
          }
          break;
        }
        case MoveToInFrontOfTarget:
        {
          ROS_INFO("MoveToInFrontOfTarget");
          tf::StampedTransform tf_transform = getTfBase(tf_listener);

          if(tf_transform.getOrigin().y() >= -1.4)
          {
            std::vector<double> positions { 0.22, -1.57, 0.0, 0.0, 0.0 };
            ros::Duration duration;
            duration.sec = 1;

            moveBase(pub_base_twist, +0.7, 0.0, 0.0, 0.0, 0.0, 0.0);
            moveArm(pub_arm_trajectory, positions, duration);
          }
          else
          {
            stopBase(pub_base_twist);

            step_++;
          }

          break;
        }*/
        case Grasp:
        {
          /*ROS_INFO("Grasp");
          tf::StampedTransform tf_transform = getTfBase(tf_listener);

          if(tf_transform.getOrigin().y() >= -1.5)
          {
            moveBase(pub_base_twist, +1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
          }
          else
          {
            *//*ADD*/
            ROS_INFO("Grasp!!!!!!!");
		  tf::StampedTransform tf_transform = getTfBase(tf_listener);
		  
		  if(faceTowardObject(tf_transform, pub_base_twist, "desk")){
			  if(moveTowardObject(tf_transform, pub_base_twist, "desk")){
				  moveBase(pub_base_twist, +1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
				 //stopBase(pub_base_twist);
				 //grasp(pub_gripper_trajectory);
			  }
		  }	
            //stopBase(pub_base_twist);
            //grasp(pub_gripper_trajectory);

            waiting_start_time = ros::Time::now();
            step_++;
          //}

          break;
        }
        case WaitForGrasping:
        {
          if(ros::Time::now() - waiting_start_time > ros::Duration(3, 0))
          {
            sendMessage(pub_msg, MSG_OBJECT_GRASPED);
            step_++;
          }

          break;
        }
        case ComeBack:
        {
          tf::StampedTransform tf_transform = getTfBase(tf_listener);

          if(tf_transform.getOrigin().y() <= 0.0)
          {
            moveBase(pub_base_twist, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
          }
          else
          {
            stopBase(pub_base_twist);

            step_++;
          }

          break;
        }
        case ReleaseObject:
        {
          openHand(pub_gripper_trajectory);
          waiting_start_time = ros::Time::now();

          step_++;

          break;
        }
        case WaitForReleasing:
        {
          if(ros::Time::now() - waiting_start_time > ros::Duration(3, 0))
          {
            sendMessage(pub_msg, MSG_TASK_FINISHED);
            step_++;
          }

          break;
        }
        case TaskFinished:
        {
          if(is_finished_)
          {
            ROS_INFO("Task finished!");
            step_ = Initialize;
          }

          break;
        }
      }

      ros::spinOnce();

      loop_rate.sleep();
    }

    return EXIT_SUCCESS;
  }
};

//add
template <class T>
T InteractiveCleanupSample::clamp(const T val, const T min, const T max)
{
  return std::min<T>(std::max<T>(min, val), max);
}
//
int main(int argc, char **argv)
{
  InteractiveCleanupSample interactive_cleanup_sample;

  return interactive_cleanup_sample.run(argc, argv);
};


