/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIV_MAPPER_H
#define LIV_MAPPER_H

#include "IMU_Processing.h"
#include "preprocess.h"
#include <image_transport/image_transport.h>
#include <nav_msgs/Path.h>
#include "voxel_map.h"  
#include <pcl/filters/voxel_grid.h>
#include "BTC.h"
class LIVMapper
{
public:
  LIVMapper(ros::NodeHandle &nh);
  ~LIVMapper();
  void initializeSubscribersAndPublishers(ros::NodeHandle &nh);
  void initializeComponents();
  void initializeFiles();
  void run();
  void gravityAlignment();
  void handleFirstFrame();
  void stateEstimationAndMapping();
  void handleLIO();
  void savePCD();
  void processImu();
  
  bool sync_packages(LidarMeasureGroup &meas);
  void prop_imu_once(StatesGroup &imu_prop_state, const double dt, V3D acc_avr, V3D angvel_avr);
  void imu_prop_callback(const ros::TimerEvent &e);
  void transformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud);
  void pointBodyToWorld(const PointType &pi, PointType &po);
 
  void RGBpointBodyToWorld(PointType const *const pi, PointType *const po);
  void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg_in);
  void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in);
  void publish_frame_world(const ros::Publisher &pubLaserCloudFullRes);
  void publish_effect_world(const ros::Publisher &pubLaserCloudEffect, const std::vector<PointToPlane> &ptpl_list);
  void publish_odometry(const ros::Publisher &pubOdomAftMapped);
  void publish_mavros(const ros::Publisher &mavros_pose_publisher);
  void publish_path(const ros::Publisher pubPath);
  void readParameters(ros::NodeHandle &nh);
  void loopDetection(
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& p,
    int id);
    void cleanupLoopDetection();
  template <typename T> void set_posestamp(T &out);
  template <typename T> void pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi, Eigen::Matrix<T, 3, 1> &po);
  template <typename T> Eigen::Matrix<T, 3, 1> pointBodyToWorld(const Eigen::Matrix<T, 3, 1> &pi);

  std::mutex mtx_buffer, mtx_buffer_imu_prop;
  std::condition_variable sig_buffer;

  SLAM_MODE slam_mode_;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map;
  
  string root_dir;
  string lid_topic, imu_topic, seq_name;
  V3D extT;
  M3D extR;

  int feats_down_size = 0, max_iterations = 0;

  double res_mean_last = 0.05;
  double gyr_cov = 0, acc_cov = 0, inv_expo_cov = 0;
  double blind_rgb_points = 0.0;
  double last_timestamp_lidar = -1.0, last_timestamp_imu = -1.0, last_timestamp_img = -1.0;
  double filter_size_surf_min = 0;
  double filter_size_pcd = 0;
  double _first_lidar_time = 0.0;
  double match_time = 0, solve_time = 0, solve_const_H_time = 0;

  bool lidar_map_inited = false, pcd_save_en = false, pub_effect_point_en = false, pose_output_en = false, ros_driver_fix_en = false, hilti_en = false;
  int pcd_save_interval = -1, pcd_index = 0;
  int pub_scan_num = 1;

  StatesGroup imu_propagate, latest_ekf_state;

  bool new_imu = false, state_update_flg = false, imu_prop_enable = true, ekf_finish_once = false;
  deque<sensor_msgs::Imu> prop_imu_buffer;
  sensor_msgs::Imu newest_imu;
  double latest_ekf_time;
  nav_msgs::Odometry imu_prop_odom;
  ros::Publisher pubImuPropOdom;
  double imu_time_offset = 0.0;
  double lidar_time_offset = 0.0;

  bool gravity_align_en = false, gravity_align_finished = false;

  bool sync_jump_flag = false;

  bool lidar_pushed = false, imu_en, gravity_est_en, flg_reset = false, ba_bg_est_en = true;
  bool dense_map_en = false;
  int  imu_int_frame = 3;
  int lidar_en = 1;
  bool is_first_frame = false;
  int grid_size, grid_n_width, grid_n_height ;

  double plot_time;
  int frame_cnt;
  deque<PointCloudXYZI::Ptr> lid_raw_data_buffer;
  deque<double> lid_header_time_buffer;
  deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
  deque<cv::Mat> img_buffer;
  deque<double> img_time_buffer;
  vector<double> extrinT;
  vector<double> extrinR;



  PointCloudXYZI::Ptr visual_sub_map;
  PointCloudXYZI::Ptr feats_undistort;
  PointCloudXYZI::Ptr feats_down_body;
  PointCloudXYZI::Ptr feats_down_world;
  PointCloudXYZI::Ptr pcl_w_wait_pub;
  PointCloudXYZI::Ptr pcl_wait_pub;
  PointCloudXYZRGB::Ptr pcl_wait_save;
  PointCloudXYZI::Ptr pcl_wait_save_intensity;

  ofstream fout_pre, fout_out, fout_pcd_pos, fout_points;

  pcl::VoxelGrid<PointType> downSizeFilterSurf;

  V3D euler_cur;

  LidarMeasureGroup LidarMeasures;
  StatesGroup _state;
  StatesGroup  state_propagat;

  nav_msgs::Path path;
  nav_msgs::Odometry odomAftMapped;
  geometry_msgs::Quaternion geoQuat;
  geometry_msgs::PoseStamped msg_body_pose;

  PreprocessPtr p_pre;
  ImuProcessPtr p_imu;
  VoxelMapManagerPtr voxelmap_manager;
  ros::Publisher plane_pub;
  ros::Publisher voxel_pub;
  ros::Subscriber sub_pcl;
  ros::Subscriber sub_imu;
  ros::Publisher pubLaserCloudFullRes;
  ros::Publisher pubNormal;
  ros::Publisher pubSubVisualMap;
  ros::Publisher pubLaserCloudEffect;
  ros::Publisher pubLaserCloudMap;
  ros::Publisher pubOdomAftMapped;
  ros::Publisher pubPath;
  ros::Publisher pubLaserCloudDyn;
  ros::Publisher pubLaserCloudDynRmed;
  ros::Publisher pubLaserCloudDynDbg;
  image_transport::Publisher pubImage;
  ros::Publisher mavros_pose_publisher;
  ros::Timer imu_prop_timer;

  int frame_num = 0;
  double aver_time_consu = 0;
  double aver_time_icp = 0;
  double aver_time_map_inre = 0;
  int global_frame_id_ = 0;

  struct LoopInputData {
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
  Eigen::Matrix3d R;
  Eigen::Vector3d p;
  int frame_id;
};
  std::thread loop_thread_;
  std::mutex loop_mtx_;
  std::condition_variable loop_cv_;
  std::queue<LoopInputData> loop_queue_;
  bool loop_running_ = true;

  // 子线程函数
void loopDetectionThread();
struct FrameData {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;  // 世界坐标系点云
    Eigen::Matrix3d R;                            // 旋转
    Eigen::Vector3d p;                            // 平移
    int id;                                       // 全局帧 ID
  };
std::deque<FrameData> frame_window_;

Eigen::Matrix3d last_kf_R_ = Eigen::Matrix3d::Identity();
Eigen::Vector3d last_kf_p_ = Eigen::Vector3d::Zero();
bool has_keyframe_ = false;           // 是否已创建过至少一个关键帧
double cumulative_distance_ = 0.0;    // 累积行驶距离
Eigen::Vector3d prev_frame_p_ = Eigen::Vector3d::Zero();  // 上一帧位置（用于累积距离）
bool has_prev_frame_ = false;
STDescManager* std_manager_ = nullptr;

  struct KeyframeInfo {
      int id;                 // 全局帧 ID
      Eigen::Matrix3d R;      // 位姿旋转
      Eigen::Vector3d p;      // 位姿平移
      double cum_dist;        // 创建时的累积行驶距离
  };
std::vector<KeyframeInfo> kf_infos_;
int keyframe_count_ = 0;

int    win_size_          = 10;     // 滑动窗口大小（几帧拼成一个关键帧）
double kf_angle_thresh_   = 5.0;    // 关键帧角度阈值（度），低于此值不创建关键帧
double kf_dist_thresh_    = 0.1;    // 关键帧距离阈值（米），低于此值不创建关键帧
double loop_score_thresh_ = 0.45;   // STD 匹配得分阈值
double icp_eigval_thresh_ = 14.0;   // ICP 特征值阈值
double ratio_drift_       = 0.05;   // 漂移比率阈值
double voxel_size_        = 0.5;    // 体素下采样大小
void initLoopDetection( ConfigSetting& config) {
      std_manager_ = new STDescManager(config);
  }
};
static Eigen::Vector3d LogSO3(const Eigen::Matrix3d& R) {
    double cos_angle = (R.trace() - 1.0) / 2.0;
    cos_angle = std::max(-1.0, std::min(1.0, cos_angle));
    double angle = std::acos(cos_angle);
 
    if (angle < 1e-10) {
        return Eigen::Vector3d::Zero();
    }
 
    Eigen::Vector3d axis;
    axis << R(2,1) - R(1,2),
            R(0,2) - R(2,0),
            R(1,0) - R(0,1);
    return axis * (angle / (2.0 * std::sin(angle)));
}
#endif