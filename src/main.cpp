#include "LIVMapper.h"

int main(int argc, char **argv)
{

  ros::init(argc, argv, "laserMapping");//初始化 ros 节点
  ros::NodeHandle nh;//获得 ros 节点的句柄，这是与 ROS 主系统进行通信的主要接口
  image_transport::ImageTransport it(nh);//是 ROS 中专门用于高效传输图像消息的工具类。 
  LIVMapper mapper(nh); // 初始化建图器（负责整个系统流程） 
  mapper.initializeSubscribersAndPublishers(nh, it);// 初始化系统的订阅器和发布器  
  //此处对订阅器和发布器初始化进行解析。主要做了以下几个事。 1、将lidar点云转换为pcl格式后，将点云（每个点均有偏移时间）及帧头时间推入对应的缓存器lid_raw_data_buffer 和 lid_header_time_buffer。
 //2、将对齐到 lidar 帧头时间的imu推入其缓存器imu_buffer，最近的imu时间戳更新last_timestamp_imu。
  //3、修正图像实际的触发时间，并将其转为cv格式分布存入对应的缓存器img_buffer，img_time_buffer。
  mapper.run();// 系统运行  return 0;
  return 0;
}