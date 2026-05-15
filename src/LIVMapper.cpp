/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/
#include "icp_normal.h"
#include "LIVMapper.h"
#include "voxel_downsample.h"
LIVMapper::LIVMapper(ros::NodeHandle &nh)//它被传入构造函数，使得 LIVMapper 类能够利用这个句柄来订阅话题（Topic）、发布消息、读取参数服务器（Parameter Server）上的配置参数。
    : extT(0, 0, 0),//(外部平移向量): 初始化为零向量 (0, 0, 0)。
      extR(M3D::Identity())//外部旋转矩阵): 初始化为单位矩阵 (Identity Matrix)。
{//成员变量的内存分配与重置 
  extrinT.assign(3, 0.0);
  extrinR.assign(9, 0.0);


  p_pre.reset(new Preprocess());
  p_imu.reset(new ImuProcess());

  readParameters(nh);
  VoxelMapConfig voxel_config;
  loadVoxelConfig(nh, voxel_config);
 //对相关点云指针进行重置

  feats_undistort.reset(new PointCloudXYZI()); //去畸变后的点云
  feats_down_body.reset(new PointCloudXYZI());//降采样后的载体系点云
  feats_down_world.reset(new PointCloudXYZI());//降采样后的世界系点云
  pcl_w_wait_pub.reset(new PointCloudXYZI());// 待发布的世界系点云
  pcl_wait_pub.reset(new PointCloudXYZI());//待发布的载体系点云
  pcl_wait_save.reset(new PointCloudXYZRGB());//待保存为 PCD 的 RGB 值点云
  pcl_wait_save_intensity.reset(new PointCloudXYZI()); //待保存为 PCD 的强度值点云
  voxelmap_manager.reset(new VoxelMapManager(voxel_config, voxel_map));//体素地图管理器 
    global_trajectory_.reset(new PointCloudXYZI());
  global_surf_map_.reset(new PointCloudXYZI());
  global_corner_map_.reset(new PointCloudXYZI());
  root_dir = ROOT_DIR;//根路径  来自CmakeLists.txt
  initializeFiles();//初始化PCD和Colmap输出的保存路径及文件，初始化状态预测及更新对应文件
  initializeComponents();//初始化组件
  path.header.stamp = ros::Time::now();
  path.header.frame_id = "camera_init";
  ConfigSetting config_setting;
  
  init_config_setting(config_setting, 0);  // nh 是你的 ros::NodeHandle，0 是 isHighFly

  printf("[Loop] === ConfigSetting ===\n");
  printf("[Loop] voxel_size: %.3f\n",        config_setting.voxel_size_);
  printf("[Loop] plane_det_thre: %.3f\n",    config_setting.plane_detection_thre_);
  printf("[Loop] candidate_num: %d\n",       config_setting.candidate_num_);
  printf("[Loop] skip_near_num: %d\n",       config_setting.skip_near_num_);
  printf("[Loop] descriptor_near_num: %f\n", config_setting.descriptor_near_num_);
  printf("[Loop] rough_dis_threshold: %.3f\n", config_setting.rough_dis_threshold_);
  printf("[Loop] similarity_threshold: %.3f\n", config_setting.similarity_threshold_);
  printf("[Loop] =======================\n");
  initLoopDetection(config_setting);
  loop_thread_ = std::thread(&LIVMapper::loopDetectionThread, this);
}

LIVMapper::~LIVMapper() {
  { std::lock_guard<std::mutex> lk(loop_mtx_); loop_running_ = false; }
    loop_cv_.notify_one();
    if (loop_thread_.joinable()) loop_thread_.join();
}

void LIVMapper::readParameters(ros::NodeHandle &nh)
{
  nh.param<string>("common/lid_topic", lid_topic, "/livox/lidar");
  nh.param<string>("common/imu_topic", imu_topic, "/livox/imu");
  nh.param<bool>("common/ros_driver_bug_fix", ros_driver_fix_en, false);
  nh.param<int>("common/lidar_en", lidar_en, 1);
  nh.param<double>("time_offset/imu_time_offset", imu_time_offset, 0.0);
  nh.param<double>("time_offset/lidar_time_offset", lidar_time_offset, 0.0);
  nh.param<bool>("uav/imu_rate_odom", imu_prop_enable, false);
  nh.param<bool>("uav/gravity_align_en", gravity_align_en, false);

  nh.param<string>("evo/seq_name", seq_name, "01");
  nh.param<bool>("evo/pose_output_en", pose_output_en, false);
  nh.param<double>("imu/gyr_cov", gyr_cov, 1.0);
  nh.param<double>("imu/acc_cov", acc_cov, 1.0);
  nh.param<int>("imu/imu_int_frame", imu_int_frame, 3);
  nh.param<bool>("imu/imu_en", imu_en, false);
  nh.param<bool>("imu/gravity_est_en", gravity_est_en, true);
  nh.param<bool>("imu/ba_bg_est_en", ba_bg_est_en, true);

  nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
  nh.param<double>("preprocess/filter_size_surf", filter_size_surf_min, 0.5);
  nh.param<bool>("preprocess/hilti_en", hilti_en, false);
  nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
  nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 6);
  nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num, 3);
  nh.param<bool>("preprocess/feature_extract_enabled", p_pre->feature_enabled, false);

  nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
  nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
  nh.param<double>("pcd_save/filter_size_pcd", filter_size_pcd, 0.5);
  nh.param<vector<double>>("extrin_calib/extrinsic_T", extrinT, vector<double>());
  nh.param<vector<double>>("extrin_calib/extrinsic_R", extrinR, vector<double>());
  nh.param<double>("debug/plot_time", plot_time, -10);
  nh.param<int>("debug/frame_cnt", frame_cnt, 6);

  nh.param<double>("publish/blind_rgb_points", blind_rgb_points, 0.01);
  nh.param<int>("publish/pub_scan_num", pub_scan_num, 1);
  nh.param<bool>("publish/pub_effect_point_en", pub_effect_point_en, false);
  nh.param<bool>("publish/dense_map_en", dense_map_en, false);

  p_pre->blind_sqr = p_pre->blind * p_pre->blind;
}
void LIVMapper::initializeComponents() 
{
  downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);//这里的体素大小是对每一帧点云体素滤波用的
  extT << VEC_FROM_ARRAY(extrinT);
  extR << MAT_FROM_ARRAY(extrinR);//将std类型的外参转为 Eigen::vector和mat类型并给voxelmap_manager

  voxelmap_manager->extT_ << VEC_FROM_ARRAY(extrinT);
  voxelmap_manager->extR_ << MAT_FROM_ARRAY(extrinR);


  p_imu->set_extrinsic(extT, extR);// lidar 到 IMU 外参
  p_imu->set_gyr_cov_scale(V3D(gyr_cov, gyr_cov, gyr_cov));//陀螺仪协方差
  p_imu->set_acc_cov_scale(V3D(acc_cov, acc_cov, acc_cov));//加速度协方差 

  p_imu->set_gyr_bias_cov(V3D(0.0001, 0.0001, 0.0001));//陀螺仪偏差的协方差
  p_imu->set_acc_bias_cov(V3D(0.0001, 0.0001, 0.0001));//加速度偏差的协方差
  p_imu->set_imu_init_frame_num(imu_int_frame);//imu初始化所需帧数

  if (!imu_en) p_imu->disable_imu();//不用 imu
  if (!gravity_est_en) p_imu->disable_gravity_est();// 不估计重力
  if (!ba_bg_est_en) p_imu->disable_bias_est();//不估计偏差


  slam_mode_ = (imu_en && lidar_en) ? ONLY_LIO : ONLY_LO;//根据可用的传感器判断 使用什么模式
}

void LIVMapper::initializeFiles() 
{

 
  
  fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), std::ios::out);
  fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), std::ios::out);//打开文件 将 LIO和VIO 状态更新之前以及更新之后的分别输出
}

void LIVMapper::initializeSubscribersAndPublishers(ros::NodeHandle &nh) 
{
  sub_pcl = p_pre->lidar_type == AVIA ? 
            nh.subscribe(lid_topic, 200000, &LIVMapper::livox_pcl_cbk, this): 
            nh.subscribe(lid_topic, 200000, &LIVMapper::standard_pcl_cbk, this);
  sub_imu = nh.subscribe(imu_topic, 200000, &LIVMapper::imu_cbk, this);
  pub_voxel_normals_ = nh.advertise<visualization_msgs::Marker>("/voxel_normals", 1);
  pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);//VIO 更新后，发布世界系的当前点云（LVIO模式为RGB点云）
  pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);//发布 LIO 更新后的位姿，由函数publish_odometry调用 
  pubPath = nh.advertise<nav_msgs::Path>("/path", 10);//发布 LIO 更新后的路径
  mavros_pose_publisher = nh.advertise<geometry_msgs::PoseStamped>("/mavros/vision_pose/pose", 10);//发布LIO更新后的位姿，用于无人机的
  pubImuPropOdom = nh.advertise<nav_msgs::Odometry>("/LIVO2/imu_propagate", 10000);//默认不会用，发布Imu频率的里程计，仅是简单的运动学方程推导出来的位姿。由标志位imu_prop_enable 控制
  imu_prop_timer = nh.createTimer(ros::Duration(0.004), &LIVMapper::imu_prop_callback, this);////一个周期为 0.004 秒（即 4 毫秒，250Hz）的定时器。每当定时器触发时，会自动调用 LIVMapper::imu_prop_callback 成员函数。

}

void LIVMapper::handleFirstFrame() 
{
  if (!is_first_frame)
  {
    //判断  是否第一帧  if (!is_first_frame)。获得第一帧激光雷达的帧头时间，将这个时间
//给imu处理器中的第一帧激光雷达时间p_imu->first_lidar_time = _first_lidar_time;  修改第一帧标志位。
    _first_lidar_time = LidarMeasures.last_lio_update_time;
    p_imu->first_lidar_time = _first_lidar_time; // Only for IMU data log
    is_first_frame = true;
    cout << "FIRST LIDAR FRAME!" << endl;
  }
}

void LIVMapper::gravityAlignment() 
{
  if (!p_imu->imu_need_init && !gravity_align_finished) 
  {
    std::cout << "Gravity Alignment Starts" << std::endl;
    V3D ez(0, 0, -1), gz(_state.gravity);
    Quaterniond G_q_I0 = Quaterniond::FromTwoVectors(gz, ez);
    M3D G_R_I0 = G_q_I0.toRotationMatrix();

    _state.pos_end = G_R_I0 * _state.pos_end;
    _state.rot_end = G_R_I0 * _state.rot_end;
    _state.vel_end = G_R_I0 * _state.vel_end;
    _state.gravity = G_R_I0 * _state.gravity;
    gravity_align_finished = true;
    std::cout << "Gravity Alignment Finished" << std::endl;
  }
}

void LIVMapper::processImu() 
{
  // double t0 = omp_get_wtime();

  p_imu->Process2(LidarMeasures, _state, feats_undistort);//调用Imu处理器的Process2函数处理激光雷达测量组并获得先验状态和去畸变点云

  if (gravity_align_en) gravityAlignment();

  state_propagat = _state;
  voxelmap_manager->state_ = _state;
  voxelmap_manager->feats_undistort_ = feats_undistort;//、记录状态，并将先验状态_state和去畸变点云feats_undistort给体素地图管理器

  // double t_prop = omp_get_wtime();

  // std::cout << "[ Mapping ] feats_undistort: " << feats_undistort->size() << std::endl;
  // std::cout << "[ Mapping ] predict cov: " << _state.cov.diagonal().transpose() << std::endl;
  // std::cout << "[ Mapping ] predict sta: " << state_propagat.pos_end.transpose() << state_propagat.vel_end.transpose() << std::endl;
}

void LIVMapper::stateEstimationAndMapping() 
{
  switch (LidarMeasures.lio_vio_flg) 
  {
    case LIO:
    case LO:
      handleLIO();
      break;
  }
}


void LIVMapper::handleLIO() 
//feats_undistort:去畸变以后的点云
//LidarMeasures：包含当前帧的时间戳范围（lidar_frame_beg_time, lidar_frame_end_time）以及相关的标志位（lio_vio_flg 确认为 LIO）。
{    
  euler_cur = RotMtoEuler(_state.rot_end);//将先验旋转状态转为欧拉角：
  fout_pre << setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
           << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
           << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << endl;//将先验状态输出fout_pre到 文件  mat_pre.txt中 
           
  if (feats_undistort->empty() || (feats_undistort == nullptr)) 
  {
    std::cout << "[ LIO ]: No point!!!" << std::endl;//检查去畸变特征点云是否有点，没有点直接退出
    return;
  }

  double t0 = omp_get_wtime();

  downSizeFilterSurf.setInputCloud(feats_undistort);
  downSizeFilterSurf.filter(*feats_down_body);//对去畸变点云进行体素滤波，
  
  double t_down = omp_get_wtime();

  feats_down_size = feats_down_body->points.size(); 
  voxelmap_manager->feats_down_body_ = feats_down_body;
  transformLidar(_state.rot_end, _state.pos_end, feats_down_body, feats_down_world);//将 lidar 系转到世界系IMU,根据 IMU 的预测，我认为这些点现在应该在世界地图的这个位置;这是一个假设的世界坐标点云
  voxelmap_manager->feats_down_world_ = feats_down_world; //把这个“假设的世界坐标点云”交给 voxelmap_manager。
  voxelmap_manager->feats_down_size_ = feats_down_size;
  
  if (!lidar_map_inited) 
  {
    lidar_map_inited = true;
    voxelmap_manager->BuildVoxelMap();///构建体素地图。
  }

  double t1 = omp_get_wtime();

  voxelmap_manager->StateEstimation(state_propagat);
  _state = voxelmap_manager->state_;
  
  double t2 = omp_get_wtime();

  if (imu_prop_enable) 
  {
    ekf_finish_once = true;
    latest_ekf_state = _state;
    latest_ekf_time = LidarMeasures.last_lio_update_time;
    state_update_flg = true;
  }

  if (pose_output_en) 
  {
    static bool pos_opend = false;//位姿文件打开标志位 静态
    static int ocount = 0;
    std::ofstream outFile, evoFile;
    if (!pos_opend) 
    {
      evoFile.open(std::string(ROOT_DIR) + "Log/result/" + seq_name + ".txt", std::ios::out);
      pos_opend = true;
      if (!evoFile.is_open()) ROS_ERROR("open fail\n");
    } 
    else 
    {
      evoFile.open(std::string(ROOT_DIR) + "Log/result/" + seq_name + ".txt", std::ios::app);
      if (!evoFile.is_open()) ROS_ERROR("open fail\n");
    }
    Eigen::Matrix4d outT;
    Eigen::Quaterniond q(_state.rot_end);
    evoFile << std::fixed;
    evoFile << LidarMeasures.last_lio_update_time << " " << _state.pos_end[0] << " " << _state.pos_end[1] << " " << _state.pos_end[2] << " "
            << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;//输出 上次更新时间（此次图像捕获时间）以及状态
  }
  
  euler_cur = RotMtoEuler(_state.rot_end);
  geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));//将状态估计后的旋转转为欧拉角euler_cur再转为四元数 geoQuat 
  publish_odometry(pubOdomAftMapped);//发布后验状态的旋转和平移

  double t3 = omp_get_wtime();

  PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI());
  transformLidar(_state.rot_end, _state.pos_end, feats_down_body, world_lidar);//声明一个世界激光雷达点云指针 并按 状态估计后的状态将去畸变的体素点云转到世界系下。 
  for (size_t i = 0; i < world_lidar->points.size(); i++) //遍历世界系下的当前帧点获得更新完不确定性点的列表
  {
    voxelmap_manager->pv_list_[i].point_w << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;//获得体素管理器中不确定性点列的世界坐标  
    M3D point_crossmat = voxelmap_manager->cross_mat_list_[i];//取出体素管理器中该点对应的反对称形式
    M3D var = voxelmap_manager->body_cov_list_[i];//取出体素管理器中载体系下该点的协方差阵
    var = (_state.rot_end * extR) * var * (_state.rot_end * extR).transpose() +
          (-point_crossmat) * _state.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose() + _state.cov.block<3, 3>(3, 3);//更新不确定性到世界系下
    voxelmap_manager->pv_list_[i].var = var;//更新到体素管理器的不确定点列表中 
  }
  voxelmap_manager->UpdateVoxelMap(voxelmap_manager->pv_list_);
  //std::cout << "[ LIO ] Update Voxel Map" << std::endl;


// ===== 回环检测 + PCD 关键帧采集 =====
  {
    // ---- (a) STD 回环用：稠密点云 ----
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_for_loop(
        new pcl::PointCloud<pcl::PointXYZI>());
    cloud_for_loop->reserve(feats_undistort->size());
    for (size_t i = 0; i < feats_undistort->size(); i++)
    {
      Eigen::Vector3d p_body(feats_undistort->points[i].x,
                              feats_undistort->points[i].y,
                              feats_undistort->points[i].z);
      Eigen::Vector3d p_world = _state.rot_end * (extR * p_body + extT) + _state.pos_end;
      pcl::PointXYZI pi;
      pi.x = p_world[0];
      pi.y = p_world[1];
      pi.z = p_world[2];
      pi.intensity = feats_undistort->points[i].intensity;
      cloud_for_loop->push_back(pi);
    }

    // ---- (b) PCD 保存用：Surf/Corner 分类（方案 B）----
    // world_lidar 是前面已经用最新 _state 算出的 feats_down_body 世界系版本
    PointCloudXYZI::Ptr frame_surf_world(new PointCloudXYZI());
    PointCloudXYZI::Ptr frame_corner_world(new PointCloudXYZI());
    frame_surf_world->reserve(world_lidar->size());
    frame_corner_world->reserve(world_lidar->size());

    const auto& useful = voxelmap_manager->useful_ptpl_;
    const int N = (int)world_lidar->points.size();
    for (int i = 0; i < N; i++) {
      const auto& src = world_lidar->points[i];
      PointType p;
      p.x = src.x; p.y = src.y; p.z = src.z;
      p.intensity = src.intensity;
      if (i < (int)useful.size() && useful[i]) {
        frame_surf_world->push_back(p);
      } else {
        frame_corner_world->push_back(p);
      }
    }

    // ---- (c) 入队 ----
    {
      std::lock_guard<std::mutex> lk(loop_mtx_);
      LoopInputData d;
      d.cloud        = cloud_for_loop;
      d.surf_world   = frame_surf_world;
      d.corner_world = frame_corner_world;
      d.R            = _state.rot_end;
      d.p            = _state.pos_end;
      d.frame_id     = global_frame_id_++;
      loop_queue_.push(std::move(d));
    }
    loop_cv_.notify_one();
  }
  

  //voxelmap_manager->manageMapMemory(
  //voxelmap_manager->current_frame_id_,
  //10000,    // 最多 1 万个体素
  //800       // 最多 800 MB
  //);
  double t4 = omp_get_wtime();

  if(voxelmap_manager->config_setting_.map_sliding_en)//判断 体素管理器的地图滑动使能清除超过某一范围内的所有体素及表索引
  {
    voxelmap_manager->mapSliding();
  }
  
  PointCloudXYZI::Ptr laserCloudFullRes(dense_map_en ? feats_undistort : feats_down_body);//根据稠密地图使能决定使用 未体素化的去畸变点 还是 体素化后的去畸变点。并获得点数
  int size = laserCloudFullRes->points.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));//根据上面选择的点云的点数，设置世界系下的点云指针 

  for (int i = 0; i < size; i++) 
  {
    RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);//、循环每个点，其实就是将点云转到世界系下
  }
  *pcl_w_wait_pub = *laserCloudWorld;//将世界系下的点云设为 待发布的世界系的点云

  publish_frame_world(pubLaserCloudFullRes);// 不使用图像则执行
  if (pub_effect_point_en) publish_effect_world(pubLaserCloudEffect, voxelmap_manager->ptpl_list_);//判断  是否发布有效点使能 if (pub_effect_point_en)。是则（发布能够匹配到平面的点，转到世界系发布）
  if (voxelmap_manager->config_setting_.is_pub_plane_map_) voxelmap_manager->pubVoxelMap();
  publish_path(pubPath);//发布路径
  publish_mavros(mavros_pose_publisher);

  frame_num++;
  aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t4 - t0) / frame_num;//计算平均耗时

  // aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + (t2 - t1) / frame_num;
  // aver_time_map_inre = aver_time_map_inre * (frame_num - 1) / frame_num + (t4 - t3) / frame_num;
  // aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + (solve_time) / frame_num;
  // aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) / frame_num + solve_const_H_time / frame_num;
  // printf("[ mapping time ]: per scan: propagation %0.6f downsample: %0.6f match: %0.6f solve: %0.6f  ICP: %0.6f  map incre: %0.6f total: %0.6f \n"
  //         "[ mapping time ]: average: icp: %0.6f construct H: %0.6f, total: %0.6f \n",
  //         t_prop - t0, t1 - t_prop, match_time, solve_time, t3 - t1, t5 - t3, t5 - t0, aver_time_icp, aver_time_const_H_time, aver_time_consu);

  // printf("\033[1;36m[ LIO mapping time ]: current scan: icp: %0.6f secs, map incre: %0.6f secs, total: %0.6f secs.\033[0m\n"
  //         "\033[1;36m[ LIO mapping time ]: average: icp: %0.6f secs, map incre: %0.6f secs, total: %0.6f secs.\033[0m\n",
  //         t2 - t1, t4 - t3, t4 - t0, aver_time_icp, aver_time_map_inre, aver_time_consu);
//   printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
//   printf("\033[1;34m|                         LIO Mapping Time                    |\033[0m\n");
//   printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
//   printf("\033[1;34m| %-29s | %-27s |\033[0m\n", "Algorithm Stage", "Time (secs)");
//   printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
//   printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "DownSample", t_down - t0);//打印降采样耗时
//   printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "ICP", t2 - t1);//ICP算法耗时
//   printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "updateVoxelMap", t4 - t3);//体素地图更新
//   printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
//   printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Current Total Time", t4 - t0);//本帧总耗时
//   printf("\033[1;36m| %-29s | %-27f |\033[0m\n", "Average Total Time", aver_time_consu);//平均耗时
//   printf("\033[1;34m+-------------------------------------------------------------+\033[0m\n");
// //打印LIO整体信息
//   // printf("measures: %zu, voxel_map: %zu\n",
//   //      LidarMeasures.measures.size(),
//   //      voxelmap_manager->voxel_map_.size());
// euler_cur = RotMtoEuler(_state.rot_end);//转为欧拉角 ，将更新后的状态保存到
// Eigen::Quaterniond q_print(_state.rot_end);
// printf("\033[1;32m[ Pose ] "
//        "pos[xyz]: %.3f %.3f %.3f | "
//        "euler[RPY deg]: %.2f %.2f %.2f | "
//        "vel[xyz]: %.3f %.3f %.3f | "
//        "quat[xyzw]: %.4f %.4f %.4f %.4f\033[0m\n",
//        _state.pos_end[0], _state.pos_end[1], _state.pos_end[2],
//        euler_cur[0], euler_cur[1], euler_cur[2],
//        _state.vel_end[0], _state.vel_end[1], _state.vel_end[2],
//        q_print.x(), q_print.y(), q_print.z(), q_print.w());
// printf("\033[1;33m[ Memory ] "
//        "voxel_map: %zu (%.1f MB) | "
//        "measures: %zu | "
//        "pcd_buf: %.1f MB | "
//        "feats_undist: %zu | "
//        "pcl_w_wait: %zu | "
//        "pcl_wait: %zu | "
//        "lid_buf: %zu | "
//        "imu_buf: %zu | "
//        "prop_imu_buf: %zu\033[0m\n",
//        voxelmap_manager->voxel_map_.size(),
//        voxelmap_manager->estimateTotalMemory() / (1024.0 * 1024.0),
//        LidarMeasures.measures.size(),
//        pcl_wait_save_intensity->size() * sizeof(PointType) / (1024.0 * 1024.0),
//        feats_undistort->size(),
//        pcl_w_wait_pub->size(),
//        pcl_wait_pub->size(),
//        lid_raw_data_buffer.size(),
//        imu_buffer.size(),
//        prop_imu_buffer.size());
// // 重力估计
// double g_norm = _state.gravity.norm();

// // 这一帧有效平面法向量的方向分布（退化检测）
// Eigen::Matrix3d normal_nnt = Eigen::Matrix3d::Zero();
// for (const auto &ptpl : voxelmap_manager->ptpl_list_)
// {
//   normal_nnt += ptpl.normal_ * ptpl.normal_.transpose();
// }
// Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_normal(normal_nnt);
// V3D normal_eig = es_normal.eigenvalues();  // 从小到大

// printf("\033[1;36m[ Diag ] "
//        "pos_z: %.3f | "
//        "gravity[xyz]: %.3f %.3f %.3f (norm %.4f) | "
//        "bias_a[xyz]: %.5f %.5f %.5f | "
//        "bias_g[xyz]: %.5f %.5f %.5f | "
//        "eff_feat: %d | "
//        "normal_eig[min mid max]: %.1f %.1f %.1f\033[0m\n",
//        _state.pos_end[2],
//        _state.gravity[0], _state.gravity[1], _state.gravity[2], g_norm,
//        _state.bias_a[0], _state.bias_a[1], _state.bias_a[2],
//        _state.bias_g[0], _state.bias_g[1], _state.bias_g[2],
//        voxelmap_manager->effct_feat_num_,
//        normal_eig[0], normal_eig[1], normal_eig[2]);
  fout_out << std::setw(20) << LidarMeasures.last_lio_update_time - _first_lidar_time << " " << euler_cur.transpose() * 57.3 << " "
            << _state.pos_end.transpose() << " " << _state.vel_end.transpose() << " " << _state.bias_g.transpose() << " "
            << _state.bias_a.transpose() << " " << V3D(_state.inv_expo_time, 0, 0).transpose() << " " << feats_undistort->points.size() << std::endl;
}

void LIVMapper::savePCD()
{
  if (!pcd_save_en) return;

  // 停回环线程，保证队列中残留帧被 finalizeKeyframe 处理完
  {
    std::lock_guard<std::mutex> lk(loop_mtx_);
    loop_running_ = false;
  }
  loop_cv_.notify_one();
  if (loop_thread_.joinable()) loop_thread_.join();

  // 分段模式：把当前段的残余写成"最后一段"
  if (pcd_save_interval > 0) {
    if (kf_in_segment_ > 0) {
      flushSegment(true);
    } else {
      std::cout << YELLOW << "[PCD] No remaining KFs in current segment, nothing to flush."
                << RESET << std::endl;
    }
    std::cout << GREEN << "[PCD] Done. Total segments: " << kf_segment_index_
              << RESET << std::endl;
    return;
  }

  // 非分段模式（interval <= 0）：一次性把累积地图写成 4 个 PCD（无编号）
  std::lock_guard<std::mutex> lk(pcd_map_mtx_);

  const std::string traj_path   = std::string(ROOT_DIR) + "Log/PCD/trajectory.pcd";
  const std::string surf_path   = std::string(ROOT_DIR) + "Log/PCD/SurfMap.pcd";
  const std::string corner_path = std::string(ROOT_DIR) + "Log/PCD/CornerMap.pcd";
  const std::string global_path = std::string(ROOT_DIR) + "Log/PCD/GlobalMap.pcd";

  pcl::PCDWriter writer;

  if (global_trajectory_ && !global_trajectory_->empty()) {
    writer.writeBinary(traj_path, *global_trajectory_);
    std::cout << GREEN << "[PCD] trajectory.pcd: "
              << global_trajectory_->size() << " kfs -> " << traj_path << RESET << std::endl;
  } else {
    std::cout << YELLOW << "[PCD] trajectory empty, skip" << RESET << std::endl;
  }

  if (global_surf_map_ && !global_surf_map_->empty()) {
    writer.writeBinary(surf_path, *global_surf_map_);
    std::cout << GREEN << "[PCD] SurfMap.pcd: "
              << global_surf_map_->size() << " pts -> " << surf_path << RESET << std::endl;
  } else {
    std::cout << YELLOW << "[PCD] surf empty, skip" << RESET << std::endl;
  }

  if (global_corner_map_ && !global_corner_map_->empty()) {
    writer.writeBinary(corner_path, *global_corner_map_);
    std::cout << GREEN << "[PCD] CornerMap.pcd: "
              << global_corner_map_->size() << " pts -> " << corner_path << RESET << std::endl;
  } else {
    std::cout << YELLOW << "[PCD] corner empty, skip" << RESET << std::endl;
  }

  PointCloudXYZI::Ptr global_map(new PointCloudXYZI());
  global_map->reserve((global_surf_map_   ? global_surf_map_->size()   : 0)
                    + (global_corner_map_ ? global_corner_map_->size() : 0));
  if (global_surf_map_)   *global_map += *global_surf_map_;
  if (global_corner_map_) *global_map += *global_corner_map_;

  if (!global_map->empty()) {
    PointCloudXYZI::Ptr global_ds(new PointCloudXYZI());
    const float leaf = (filter_size_pcd > 0.01) ? (float)filter_size_pcd : 0.15f;
    voxelFilterManual(global_map, global_ds, leaf);
    writer.writeBinary(global_path, *global_ds);
    std::cout << GREEN << "[PCD] GlobalMap.pcd: "
              << global_ds->size() << " pts (merged) -> " << global_path << RESET << std::endl;
  } else {
    std::cout << YELLOW << "[PCD] global empty, skip" << RESET << std::endl;
  }
}
void LIVMapper::run() 
{
  ros::Rate rate(5000);//设置循环体的运行频率
  while (ros::ok()) //按设置频率进行循环
  {
    ros::spinOnce();//回调函数队列触发 ，检查是否有新的消息到达，并触发订阅器相应的回调函数。运行至此
//处，激光雷达IMU，相机的数据均保存至，lid_raw_data_buffer，imu_buffer，
//img_buffer 需要注意的是，spinOnce()在执行的那一刻，并不是只处理队列中的一个回
//调函数就返回了，而是会处理当时队列中存在的所有回调函数。 
    if (!sync_packages(LidarMeasures)) //将数据缓存器中同一时间区间的lidar、Imu以及image都取出，用于后续状态估计更新
    {
      //std::cout<<LidarMeasures.measures.size()<<std::endl;
      rate.sleep();
      continue;
    }
    handleFirstFrame();//记录第一帧lidar点云的帧头时间

    processImu();//进行 ESIKF 的预测（前向传播）和点云运动去畸变（后向传播）

    // if (!p_imu->imu_time_init) continue;

    stateEstimationAndMapping();//顺序执行 LIO 和VIO状态更新
  }
  savePCD();
}

void LIVMapper::prop_imu_once(StatesGroup &imu_prop_state, const double dt, V3D acc_avr, V3D angvel_avr)
{
  double mean_acc_norm = p_imu->IMU_mean_acc_norm;
  acc_avr = acc_avr * G_m_s2 / mean_acc_norm - imu_prop_state.bias_a;
  angvel_avr -= imu_prop_state.bias_g;

  M3D Exp_f = Exp(angvel_avr, dt);
  /* propogation of IMU attitude */
  imu_prop_state.rot_end = imu_prop_state.rot_end * Exp_f;

  /* Specific acceleration (global frame) of IMU */
  V3D acc_imu = imu_prop_state.rot_end * acc_avr + V3D(imu_prop_state.gravity[0], imu_prop_state.gravity[1], imu_prop_state.gravity[2]);

  /* propogation of IMU */
  imu_prop_state.pos_end = imu_prop_state.pos_end + imu_prop_state.vel_end * dt + 0.5 * acc_imu * dt * dt;

  /* velocity of IMU */
  imu_prop_state.vel_end = imu_prop_state.vel_end + acc_imu * dt;
}

void LIVMapper::imu_prop_callback(const ros::TimerEvent &e)
{
  if (p_imu->imu_need_init || !new_imu || !ekf_finish_once) { return; }
  mtx_buffer_imu_prop.lock();
  new_imu = false; // 控制propagate频率和IMU频率一致
  if (imu_prop_enable && !prop_imu_buffer.empty())
  {
    static double last_t_from_lidar_end_time = 0;
    if (state_update_flg)
    {
      imu_propagate = latest_ekf_state;
      // drop all useless imu pkg
      while ((!prop_imu_buffer.empty() && prop_imu_buffer.front().header.stamp.toSec() < latest_ekf_time))
      {
        prop_imu_buffer.pop_front();
      }
      last_t_from_lidar_end_time = 0;
      for (int i = 0; i < prop_imu_buffer.size(); i++)
      {
        double t_from_lidar_end_time = prop_imu_buffer[i].header.stamp.toSec() - latest_ekf_time;
        double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
        // cout << "prop dt" << dt << ", " << t_from_lidar_end_time << ", " << last_t_from_lidar_end_time << endl;
        V3D acc_imu(prop_imu_buffer[i].linear_acceleration.x, prop_imu_buffer[i].linear_acceleration.y, prop_imu_buffer[i].linear_acceleration.z);
        V3D omg_imu(prop_imu_buffer[i].angular_velocity.x, prop_imu_buffer[i].angular_velocity.y, prop_imu_buffer[i].angular_velocity.z);
        prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
        last_t_from_lidar_end_time = t_from_lidar_end_time;
      }
      state_update_flg = false;
    }
    else
    {
      V3D acc_imu(newest_imu.linear_acceleration.x, newest_imu.linear_acceleration.y, newest_imu.linear_acceleration.z);
      V3D omg_imu(newest_imu.angular_velocity.x, newest_imu.angular_velocity.y, newest_imu.angular_velocity.z);
      double t_from_lidar_end_time = newest_imu.header.stamp.toSec() - latest_ekf_time;
      double dt = t_from_lidar_end_time - last_t_from_lidar_end_time;
      prop_imu_once(imu_propagate, dt, acc_imu, omg_imu);
      last_t_from_lidar_end_time = t_from_lidar_end_time;
    }

    V3D posi, vel_i;
    Eigen::Quaterniond q;
    posi = imu_propagate.pos_end;
    vel_i = imu_propagate.vel_end;
    q = Eigen::Quaterniond(imu_propagate.rot_end);
    imu_prop_odom.header.frame_id = "world";
    imu_prop_odom.header.stamp = newest_imu.header.stamp;
    imu_prop_odom.pose.pose.position.x = posi.x();
    imu_prop_odom.pose.pose.position.y = posi.y();
    imu_prop_odom.pose.pose.position.z = posi.z();
    imu_prop_odom.pose.pose.orientation.w = q.w();
    imu_prop_odom.pose.pose.orientation.x = q.x();
    imu_prop_odom.pose.pose.orientation.y = q.y();
    imu_prop_odom.pose.pose.orientation.z = q.z();
    imu_prop_odom.twist.twist.linear.x = vel_i.x();
    imu_prop_odom.twist.twist.linear.y = vel_i.y();
    imu_prop_odom.twist.twist.linear.z = vel_i.z();
    pubImuPropOdom.publish(imu_prop_odom);
  }
  mtx_buffer_imu_prop.unlock();
}

void LIVMapper::transformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud)
{
  PointCloudXYZI().swap(*trans_cloud);
  trans_cloud->reserve(input_cloud->size());
  for (size_t i = 0; i < input_cloud->size(); i++)
  {
    pcl::PointXYZINormal p_c = input_cloud->points[i];
    Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
    p = (rot * (extR * p + extT) + t);
    PointType pi;
    pi.x = p(0);
    pi.y = p(1);
    pi.z = p(2);
    pi.intensity = p_c.intensity;
    trans_cloud->points.push_back(pi);
  }
}

void LIVMapper::pointBodyToWorld(const PointType &pi, PointType &po)
{
  V3D p_body(pi.x, pi.y, pi.z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po.x = p_global(0);
  po.y = p_global(1);
  po.z = p_global(2);
  po.intensity = pi.intensity;
}

template <typename T> void LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
  V3D p_body(pi[0], pi[1], pi[2]);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po[0] = p_global(0);
  po[1] = p_global(1);
  po[2] = p_global(2);
}

template <typename T> Matrix<T, 3, 1> LIVMapper::pointBodyToWorld(const Matrix<T, 3, 1> &pi)
{
  V3D p(pi[0], pi[1], pi[2]);
  p = (_state.rot_end * (extR * p + extT) + _state.pos_end);
  Matrix<T, 3, 1> po(p[0], p[1], p[2]);
  return po;
}

void LIVMapper::RGBpointBodyToWorld(PointType const *const pi, PointType *const po)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(_state.rot_end * (extR * p_body + extT) + _state.pos_end);
  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void LIVMapper::standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
  if (!lidar_en) return;
  mtx_buffer.lock();

  double cur_head_time = msg->header.stamp.toSec() + lidar_time_offset;
  // cout<<"got feature"<<endl;
  if (cur_head_time < last_timestamp_lidar)
  {
    ROS_ERROR("lidar loop back, clear buffer");
    lid_raw_data_buffer.clear();
  }
  // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());//在堆上分配内存，创建一个新的点云对象
  p_pre->process(msg, ptr);
  lid_raw_data_buffer.push_back(ptr);
  lid_header_time_buffer.push_back(cur_head_time);
  last_timestamp_lidar = cur_head_time;

  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void LIVMapper::livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg_in)
{
  if (!lidar_en) return;
  mtx_buffer.lock();
  livox_ros_driver::CustomMsg::Ptr msg(new livox_ros_driver::CustomMsg(*msg_in));//重置一个新的avia自定义消息
  // if ((abs(msg->header.stamp.toSec() - last_timestamp_lidar) > 0.2 && last_timestamp_lidar > 0) || sync_jump_flag)
  // {
  //   ROS_WARN("lidar jumps %.3f\n", msg->header.stamp.toSec() - last_timestamp_lidar);
  //   sync_jump_flag = true;
  //   msg->header.stamp = ros::Time().fromSec(last_timestamp_lidar + 0.1);
  // }
  if (abs(last_timestamp_imu - msg->header.stamp.toSec()) > 1.0 && !imu_buffer.empty())
  {
    double timediff_imu_wrt_lidar = last_timestamp_imu - msg->header.stamp.toSec();
    printf("\033[95mSelf sync IMU and LiDAR, HARD time lag is %.10lf \n\033[0m", timediff_imu_wrt_lidar - 0.100);
    // imu_time_offset = timediff_imu_wrt_lidar;如果两者相差超过 1.0秒，说明硬件时间同步出现了严重问题，或者某个传感器延时极高。
  }

  double cur_head_time = msg->header.stamp.toSec();//记录当前雷达时间戳并打印出来
  ROS_INFO("Get LiDAR, its header time: %.6f", cur_head_time);
  if (cur_head_time < last_timestamp_lidar)
  {
    ROS_ERROR("lidar loop back, clear buffer");
    lid_raw_data_buffer.clear();//判断雷达是否出现时间戳回滚，清空激光雷达数据缓存器
  }
  // ROS_INFO("get point cloud at time: %.6f", msg->header.stamp.toSec());
  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);

  if (!ptr || ptr->empty()) {
    ROS_ERROR("Received an empty point cloud");
    mtx_buffer.unlock();
    return;//如果点云没有则解开互斥锁，之间返回 空
  }

  lid_raw_data_buffer.push_back(ptr);//将当前帧处理后的点云 推入 雷达原始数据缓存器
  lid_header_time_buffer.push_back(cur_head_time);//将每帧点云的帧头时间都推入雷达帧头时间缓存器 
  last_timestamp_lidar = cur_head_time;

  mtx_buffer.unlock();
  sig_buffer.notify_all();//解开互斥锁并通知所有线程
}

void LIVMapper::imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
  if (!imu_en) return;

  if (last_timestamp_lidar < 0.0) return;//最近的雷达时间戳小于0 ，等待雷达
  // ROS_INFO("get imu at time: %.6f", msg_in->header.stamp.toSec());
  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));//创建副本，确保线程安全。
  msg->header.stamp = ros::Time().fromSec(msg->header.stamp.toSec() - imu_time_offset);//将Imu时间戳对齐到lidar时间戳，这是硬编码或参数配置的硬件延时补偿如果已知 IMU 比 LiDAR 慢 10ms，就在这里手动减去 0.01s，强行对齐时间轴
  double timestamp = msg->header.stamp.toSec();//记录时间戳

  if (fabs(last_timestamp_lidar - timestamp) > 0.5 && (!ros_driver_fix_en))//如果当前 IMU 时间和最近一次 LiDAR 时间相差超过 0.5秒，报黄警。
  {
    ROS_WARN("IMU and LiDAR not synced! delta time: %lf .\n", last_timestamp_lidar - timestamp);
  }

  if (ros_driver_fix_en) timestamp += std::round(last_timestamp_lidar - timestamp);
  msg->header.stamp = ros::Time().fromSec(timestamp);//不用管，默认是false

  mtx_buffer.lock();//时间没问题后 进行互斥锁

  if (last_timestamp_imu > 0.0 && timestamp < last_timestamp_imu)//imu时间回滚。当前时间戳小于上一个时间戳
  {
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    ROS_ERROR("imu loop back, offset: %lf \n", last_timestamp_imu - timestamp);//它没有清空 Buffer，而是直接丢弃当前帧
    return;
  }

  // if (last_timestamp_imu > 0.0 && timestamp > last_timestamp_imu + 0.2)
  // {

  //   ROS_WARN("imu time stamp Jumps %0.4lf seconds \n", timestamp - last_timestamp_imu);
  //   mtx_buffer.unlock();
  //   sig_buffer.notify_all();
  //   return;
  // }

  last_timestamp_imu = timestamp;
  imu_buffer.push_back(msg);
  // cout<<"got imu: "<<timestamp<<" imu size "<<imu_buffer.size()<<endl;
  mtx_buffer.unlock();//更新上个IMU时间戳，并推入imu消息到imu缓存器中。解开互斥

  if (imu_prop_enable)
  {
    mtx_buffer_imu_prop.lock();
    if (imu_prop_enable && !p_imu->imu_need_init) { prop_imu_buffer.push_back(*msg); }
    newest_imu = *msg;//缓存最新一帧
    new_imu = true;
    mtx_buffer_imu_prop.unlock();
  }
  sig_buffer.notify_all();
}

bool LIVMapper::sync_packages(LidarMeasureGroup &meas)
{
  if (lid_raw_data_buffer.empty() && lidar_en) return false;
  if (imu_buffer.empty() && imu_en) return false;//判断使用的传感器缓存区是否有数据，否则直接返回false

  switch (slam_mode_)
  {
  case ONLY_LIO:
  {
    if (meas.last_lio_update_time < 0.0) meas.last_lio_update_time = lid_header_time_buffer.front();
    if (!lidar_pushed)//判断一帧点云是否被推入到了激光雷达测量组
    {
      // If not push the lidar into measurement data buffer
      meas.lidar = lid_raw_data_buffer.front(); // push the first lidar topic
      if (meas.lidar->points.size() <= 1) return false;

      meas.lidar_frame_beg_time = lid_header_time_buffer.front();                                                // generate lidar_frame_beg_time
      meas.lidar_frame_end_time = meas.lidar_frame_beg_time + meas.lidar->points.back().curvature / double(1000); // calc lidar scan end time
      //std::cout<<"lidar_frame_beg_time:"<<meas.lidar_frame_beg_time<<"     lidar_frame_end_time:"<<meas.lidar_frame_end_time<<std::endl;
      meas.pcl_proc_cur= meas.lidar;
      lidar_pushed = true;                                                                                       // flag
    }

    if (imu_en && last_timestamp_imu < meas.lidar_frame_end_time)
    { 
    // 时间轴: ------------------------------------> t
    // 雷达帧:      [Start ............... End]
    //                                      ^
    // IMU流 :      [x-x-x-x-x-x-x-x-x]     |
    //                                ^     |
    //                     last_timestamp_imu < End
      return false;
    }

    struct MeasureGroup m; // standard method to keep imu message.

    m.imu.clear();
    m.lio_time = meas.lidar_frame_end_time;//清空测量组的imu队列，获得测量组的lio开始时间戳为最后一个激光点时间
    meas.measures.clear();
    mtx_buffer.lock();
    while (!imu_buffer.empty())
    {
      if (imu_buffer.front()->header.stamp.toSec() > meas.lidar_frame_end_time) break;
      m.imu.push_back(imu_buffer.front());
      imu_buffer.pop_front();
    }
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();

    meas.lio_vio_flg = LIO; // process lidar topic, so timestamp should be lidar scan end.
    meas.measures.push_back(m);
    // ROS_INFO("ONlY HAS LiDAR and IMU, NO IMAGE!");
    lidar_pushed = false; // sync one whole lidar scan.
    return true;

    break;
  }


  case ONLY_LO:
  {
    if (!lidar_pushed) 
    { 
      // If not in lidar scan, need to generate new meas
      if (lid_raw_data_buffer.empty())  return false;
      meas.lidar = lid_raw_data_buffer.front(); // push the first lidar topic
      meas.lidar_frame_beg_time = lid_header_time_buffer.front(); // generate lidar_beg_time
      meas.lidar_frame_end_time  = meas.lidar_frame_beg_time + meas.lidar->points.back().curvature / double(1000); // calc lidar scan end time
      lidar_pushed = true;             
    }
    struct MeasureGroup m; // standard method to keep imu message.
    m.lio_time = meas.lidar_frame_end_time;
    mtx_buffer.lock();
    lid_raw_data_buffer.pop_front();
    lid_header_time_buffer.pop_front();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    lidar_pushed = false; // sync one whole lidar scan.
    meas.lio_vio_flg = LO; // process lidar topic, so timestamp should be lidar scan end.
    meas.measures.push_back(m);
    return true;
    break;
  }

  default:
  {
    printf("!! WRONG SLAM TYPE !!");
    return false;
  }
  }
  ROS_ERROR("out sync");
}
void LIVMapper::publish_frame_world(const ros::Publisher &pubLaserCloudFullRes)
{
  if (pcl_w_wait_pub->empty()) return;

  sensor_msgs::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*pcl_w_wait_pub, laserCloudmsg);
  laserCloudmsg.header.stamp = ros::Time::now();
  laserCloudmsg.header.frame_id = "camera_init";
  pubLaserCloudFullRes.publish(laserCloudmsg);



  pcl_w_wait_pub->clear();
  pcl_wait_pub->clear();
}



void LIVMapper::publish_effect_world(const ros::Publisher &pubLaserCloudEffect, const std::vector<PointToPlane> &ptpl_list)
{
  int effect_feat_num = ptpl_list.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effect_feat_num, 1));
  for (int i = 0; i < effect_feat_num; i++)
  {
    laserCloudWorld->points[i].x = ptpl_list[i].point_w_[0];
    laserCloudWorld->points[i].y = ptpl_list[i].point_w_[1];
    laserCloudWorld->points[i].z = ptpl_list[i].point_w_[2];
  }
  sensor_msgs::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
  laserCloudFullRes3.header.stamp = ros::Time::now();
  laserCloudFullRes3.header.frame_id = "camera_init";
  pubLaserCloudEffect.publish(laserCloudFullRes3);
}

template <typename T> void LIVMapper::set_posestamp(T &out)
{
  out.position.x = _state.pos_end(0);
  out.position.y = _state.pos_end(1);
  out.position.z = _state.pos_end(2);
  out.orientation.x = geoQuat.x;
  out.orientation.y = geoQuat.y;
  out.orientation.z = geoQuat.z;
  out.orientation.w = geoQuat.w;
}

void LIVMapper::publish_odometry(const ros::Publisher &pubOdomAftMapped)
{
  odomAftMapped.header.frame_id = "camera_init";
  odomAftMapped.child_frame_id = "aft_mapped";
  odomAftMapped.header.stamp = ros::Time::now(); //.ros::Time()fromSec(last_timestamp_lidar);
  set_posestamp(odomAftMapped.pose.pose);

  static tf::TransformBroadcaster br;
  tf::Transform transform;
  tf::Quaternion q;
  transform.setOrigin(tf::Vector3(_state.pos_end(0), _state.pos_end(1), _state.pos_end(2)));
  q.setW(geoQuat.w);
  q.setX(geoQuat.x);
  q.setY(geoQuat.y);
  q.setZ(geoQuat.z);
  transform.setRotation(q);
  br.sendTransform( tf::StampedTransform(transform, odomAftMapped.header.stamp, "camera_init", "aft_mapped") );
  pubOdomAftMapped.publish(odomAftMapped);
}

void LIVMapper::publish_mavros(const ros::Publisher &mavros_pose_publisher)
{
  msg_body_pose.header.stamp = ros::Time::now();
  msg_body_pose.header.frame_id = "camera_init";
  set_posestamp(msg_body_pose.pose);
  mavros_pose_publisher.publish(msg_body_pose);
}

void LIVMapper::publish_path(const ros::Publisher pubPath)
{
  set_posestamp(msg_body_pose.pose);
  msg_body_pose.header.stamp = ros::Time::now();
  msg_body_pose.header.frame_id = "camera_init";
  path.poses.push_back(msg_body_pose);
  pubPath.publish(path);
}
void LIVMapper::loopDetectionThread() {
    while (true) {
        std::unique_lock<std::mutex> lk(loop_mtx_);
        loop_cv_.wait(lk, [&]{ return !loop_queue_.empty() || !loop_running_; });
        if (!loop_running_ && loop_queue_.empty()) break;
        LoopInputData d = std::move(loop_queue_.front());
        loop_queue_.pop();
        lk.unlock();
        loopDetection(d.cloud, d.R, d.p, d.frame_id,
                      d.surf_world, d.corner_world);
    }
}

void LIVMapper::loopDetection(
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& p,
    int id,
    PointCloudXYZI::Ptr surf_world,
    PointCloudXYZI::Ptr corner_world)
{
    // ====================================================================
    // Step 1: 累积行驶距离
    // ====================================================================

        if (!cloud || cloud->empty()) {
        printf("[Loop] Skip: empty cloud (id=%d)\n", id);
        return;
    }



    if (has_prev_frame_) {
        cumulative_distance_ += (p - prev_frame_p_).norm();
    }
    prev_frame_p_ = p;
    has_prev_frame_ = true;
 
    // ====================================================================
    // Step 2: 将当前帧加入滑动窗口
    // ====================================================================
 
    FrameData fd;
    fd.cloud        = cloud;
    fd.surf_world   = surf_world;
    fd.corner_world = corner_world;
    fd.R = R;
    fd.p = p;
    fd.id = id;
    frame_window_.push_back(fd);
 
    // ====================================================================
    // Step 3: 窗口未满，等待更多帧
    // ====================================================================

 
    if ((int)frame_window_.size() < win_size_) {
        return;
    }
 
    // ====================================================================
    // Step 4: 关键帧判定 —— 运动变化是否足够大
    // ====================================================================

 
    if (has_keyframe_) {
        Eigen::Matrix3d delta_R = last_kf_R_.transpose() * R;
        double angle_deg = LogSO3(delta_R).norm() * 57.2957795;  // 弧度 → 度
        double dist = (p - last_kf_p_).norm();
 
        if (angle_deg < kf_angle_thresh_ && dist < kf_dist_thresh_) {
            // 运动太小，丢弃窗口最早帧，窗口前移一格，等下一帧再判断
            frame_window_.pop_front();
            return;
        }
    }
 
    // ====================================================================
    // Step 5: 关键帧点云构建 —— 多帧拼接到当前帧局部坐标系
    // ====================================================================

    const Eigen::Matrix3d& R_ref = R;   // 参考系旋转 = 当前帧旋转
    const Eigen::Vector3d& p_ref = p;   // 参考系原点 = 当前帧位置
 
    pcl::PointCloud<pcl::PointXYZI>::Ptr merged_cloud(
        new pcl::PointCloud<pcl::PointXYZI>());
 
    for (int i = 0; i < win_size_; i++) {
        const FrameData& frame = frame_window_[i];

        Eigen::Matrix3d delta_R = R_ref.transpose() * frame.R;
        Eigen::Vector3d delta_p = R_ref.transpose() * (frame.p - p_ref);
 
        for (const auto& pt : frame.cloud->points) {

            Eigen::Vector3d pw(pt.x, pt.y, pt.z);
            Eigen::Vector3d pl = R_ref.transpose() * (pw - p_ref);
 
            pcl::PointXYZI pt_local;
            pt_local.x = pl[0];
            pt_local.y = pl[1];
            pt_local.z = pl[2];
            pt_local.intensity = pt.intensity;
            merged_cloud->push_back(pt_local);
        }
    }
 
    // ====================================================================
    // Step 6: 清空滑动窗口
    // ====================================================================

 
    // ====================================================================
    // Step 6a: 聚合当前窗口的 Surf/Corner（世界系），为 PCD 保存准备
    // ====================================================================
    PointCloudXYZI::Ptr kf_surf(new PointCloudXYZI());
    PointCloudXYZI::Ptr kf_corner(new PointCloudXYZI());
    for (const auto& fr : frame_window_) {
      if (fr.surf_world   && !fr.surf_world->empty())   *kf_surf   += *fr.surf_world;
      if (fr.corner_world && !fr.corner_world->empty()) *kf_corner += *fr.corner_world;
    }

    // ====================================================================
    // Step 6b: 清空滑动窗口
    // ====================================================================
    frame_window_.clear();
 
    // ====================================================================
    // Step 7: 下采样
    // ====================================================================

pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_down(
    new pcl::PointCloud<pcl::PointXYZI>());
double leaf = voxel_size_ / 10.0;
voxel_downsample(*merged_cloud, *cloud_down, leaf);
 
    // ====================================================================
    // Step 8: 提取 STD 描述子
    // ====================================================================

    // printf("[Loop] Before GenerateSTDescs: cloud_down size=%lu, keyframe_count=%d, std_manager=%p\n",
    //    cloud_down->size(), keyframe_count_, (void*)std_manager_);
    std::vector<STD> stds_vec;
    std_manager_->GenerateSTDescs(cloud_down, stds_vec, keyframe_count_);
 
    // ====================================================================
    // Step 9: 回环搜索 + ICP 验证 + 漂移检查
    // ====================================================================
 
    if (keyframe_count_ > 0) {
        // --- 9a: STD 描述子搜索 ---
 
        std::pair<int, double> search_result(-1, 0);
        // .first  = 匹配到的关键帧在 plane_cloud_vec_ 中的索引（-1 = 无匹配）
        // .second = 匹配得分
 
        std::pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform;
        // .first  = 相对平移
        // .second = 相对旋转
 
        std::vector<std::pair<STD, STD>> loop_std_pair;
        // 匹配的描述子对（仅调试用）
 
        std_manager_->SearchLoop(
            stds_vec,                                      // 当前关键帧的描述子
            search_result,                                 // [输出] 匹配结果
            loop_transform,                                // [输出] 相对变换
            loop_std_pair,                                 // [输出] 描述子对
            std_manager_->plane_cloud_vec_.back());        // 当前关键帧的平面点云
 
        if (search_result.first >= 0) {
            printf("[Loop] STD candidate: current_kf=%d, matched_kf=%d, score=%.3f\n",
                   keyframe_count_, search_result.first, search_result.second);
        }
 
        // --- 9b: 得分超过阈值 → ICP 验证 ---
 
        if (search_result.first >= 0 && search_result.second > loop_score_thresh_) {
 
            bool icp_ok = icp_normal(
                *(std_manager_->plane_cloud_vec_.back()),                 // 当前帧平面点云
                *(std_manager_->plane_cloud_vec_[search_result.first]),   // 匹配帧平面点云
                loop_transform,                                           // 相对变换（被精化）
                icp_eigval_thresh_);                                      // 特征值阈值
 
            if (icp_ok) {
                // --- 9c: ICP 通过，执行漂移比率检查 ---
 
                // 从 plane_cloud_vec_ 的 header.seq 中取出匹配帧对应的关键帧编号
                int matched_kf_idx =
                    std_manager_->plane_cloud_vec_[search_result.first]->header.seq;
 
                if (matched_kf_idx >= 0 && matched_kf_idx < (int)kf_infos_.size()) {
 
                    const KeyframeInfo& matched_kf = kf_infos_[matched_kf_idx];
 
                    // 计算位姿漂移:
                    //   通过回环变换推算当前帧位置: matched_R * relative_p + matched_p
                    //   与当前帧里程计位置 p 做差
                    double drift_p = (matched_kf.R * loop_transform.first
                                      + matched_kf.p - p).norm();
 
                    // 两帧之间实际行驶的距离
                    double travel_span = cumulative_distance_ - matched_kf.cum_dist;
 
                    printf("[Loop] ICP passed. drift=%.3fm, span=%.3fm, ratio=%.4f\n",
                           drift_p, travel_span,
                           (travel_span > 1e-6) ? drift_p / travel_span : 999.0);
 
                    // 漂移比率检查:
                    //   drift / travel_span < ratio_drift_ 才接受
                    //   直觉: 走了 100m 漂移 3m → 0.03 < 0.05 → 合理
                    //         走了 5m 漂移 3m   → 0.60 > 0.05 → 不合理（可能误匹配）
                    if (travel_span > 1e-3 && drift_p / travel_span < ratio_drift_) {
                        printf("[Loop] ===== LOOP DETECTED =====\n");
                        printf("[Loop]   current  kf: %d  (id=%d, pos=[%.2f, %.2f, %.2f])\n",
                               keyframe_count_, id, p[0], p[1], p[2]);
                        printf("[Loop]   matched  kf: %d  (id=%d, pos=[%.2f, %.2f, %.2f])\n",
                               matched_kf_idx, matched_kf.id,
                               matched_kf.p[0], matched_kf.p[1], matched_kf.p[2]);
                        printf("[Loop]   score=%.3f, drift=%.3fm\n",
                               search_result.second, drift_p);
                        printf("[Loop] =========================\n");
                    } else {
                        printf("[Loop] Rejected: drift ratio too high\n");
                    }
 
                } else {
                    printf("[Loop] Warning: matched_kf_idx=%d out of range\n", matched_kf_idx);
                }
 
            } else {
                printf("[Loop] ICP verification failed\n");
            }
        }
    }
 
    // ====================================================================
    // Step 10: 将当前帧的描述子加入数据库 & 记录关键帧信息
    // ====================================================================
 
    // 描述子入库: 后续帧就能和当前关键帧做匹配了
    std_manager_->AddSTDescs(stds_vec);
 
    // 记录关键帧信息（位姿 + 累积距离），用于后续的漂移比率检查
    KeyframeInfo info;
    info.id       = id;
    info.R        = R;
    info.p        = p;
    info.cum_dist = cumulative_distance_;
    kf_infos_.push_back(info);
 
    // 更新上一关键帧位姿（下次 Step 4 要用）
    last_kf_R_ = R;
    last_kf_p_ = p;
    has_keyframe_ = true;
 
    finalizeKeyframe(keyframe_count_, kf_surf, kf_corner, p);

    keyframe_count_++;
 
    // printf("[Loop] Keyframe %d created (id=%d, pos=[%.2f, %.2f, %.2f], total_dist=%.2fm)\n",
    //        keyframe_count_ - 1, id, p[0], p[1], p[2], cumulative_distance_);
}
void LIVMapper::finalizeKeyframe(int kf_idx,
                                 const PointCloudXYZI::Ptr& kf_surf,
                                 const PointCloudXYZI::Ptr& kf_corner,
                                 const Eigen::Vector3d& kf_p)
{
  // 关键帧级别再降一次采样（基于 filter_size_pcd，默认 0.15）
  const float leaf = (filter_size_pcd > 0.01) ? (float)filter_size_pcd : 0.15f;

  PointCloudXYZI::Ptr kf_surf_ds(new PointCloudXYZI());
  PointCloudXYZI::Ptr kf_corner_ds(new PointCloudXYZI());
  if (kf_surf)   voxelFilterManual(kf_surf,   kf_surf_ds,   leaf);
  if (kf_corner) voxelFilterManual(kf_corner, kf_corner_ds, leaf);

  // intensity = 关键帧索引
  for (auto& pt : kf_surf_ds->points)   pt.intensity = (float)kf_idx;
  for (auto& pt : kf_corner_ds->points) pt.intensity = (float)kf_idx;

  // trajectory 点
  PointType tp;
  tp.x = kf_p[0];
  tp.y = kf_p[1];
  tp.z = kf_p[2];
  tp.intensity = (float)kf_idx;

  {
    std::lock_guard<std::mutex> lk(pcd_map_mtx_);
    *global_surf_map_   += *kf_surf_ds;
    *global_corner_map_ += *kf_corner_ds;
    global_trajectory_->push_back(tp);
    kf_in_segment_++;
  }
  euler_cur = RotMtoEuler(_state.rot_end);//转为欧拉角 ，将更新后的状态保存到
Eigen::Quaterniond q_print(_state.rot_end);
printf("\033[1;32m[ Pose ] "
       "pos[xyz]: %.3f %.3f %.3f | "
       "euler[RPY deg]: %.2f %.2f %.2f | "
       "vel[xyz]: %.3f %.3f %.3f | "
       "quat[xyzw]: %.4f %.4f %.4f %.4f\033[0m\n",
       _state.pos_end[0], _state.pos_end[1], _state.pos_end[2],
       euler_cur[0], euler_cur[1], euler_cur[2],
       _state.vel_end[0], _state.vel_end[1], _state.vel_end[2],
       q_print.x(), q_print.y(), q_print.z(), q_print.w());
printf("\033[1;33m[ Memory ] "
       "voxel_map: %zu (%.1f MB) | "
       "measures: %zu | "
       "pcd_buf: %.1f MB | "
       "feats_undist: %zu | "
       "pcl_w_wait: %zu | "
       "pcl_wait: %zu | "
       "lid_buf: %zu | "
       "imu_buf: %zu | "
       "prop_imu_buf: %zu\033[0m\n",
       voxelmap_manager->voxel_map_.size(),
       voxelmap_manager->estimateTotalMemory() / (1024.0 * 1024.0),
       LidarMeasures.measures.size(),
       pcl_wait_save_intensity->size() * sizeof(PointType) / (1024.0 * 1024.0),
       feats_undistort->size(),
       pcl_w_wait_pub->size(),
       pcl_wait_pub->size(),
       lid_raw_data_buffer.size(),
       imu_buffer.size(),
       prop_imu_buffer.size());
// 重力估计
double g_norm = _state.gravity.norm();

// 这一帧有效平面法向量的方向分布（退化检测）
Eigen::Matrix3d normal_nnt = Eigen::Matrix3d::Zero();
for (const auto &ptpl : voxelmap_manager->ptpl_list_)
{
  normal_nnt += ptpl.normal_ * ptpl.normal_.transpose();
}
Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_normal(normal_nnt);
V3D normal_eig = es_normal.eigenvalues();  // 从小到大

printf("\033[1;36m[ Diag ] "
       "pos_z: %.3f | "
       "gravity[xyz]: %.3f %.3f %.3f (norm %.4f) | "
       "bias_a[xyz]: %.5f %.5f %.5f | "
       "bias_g[xyz]: %.5f %.5f %.5f | "
       "eff_feat: %d | "
       "normal_eig[min mid max]: %.1f %.1f %.1f\033[0m\n",
       _state.pos_end[2],
       _state.gravity[0], _state.gravity[1], _state.gravity[2], g_norm,
       _state.bias_a[0], _state.bias_a[1], _state.bias_a[2],
       _state.bias_g[0], _state.bias_g[1], _state.bias_g[2],
       voxelmap_manager->effct_feat_num_,
       normal_eig[0], normal_eig[1], normal_eig[2]);
  // printf("\033[1;32m[PCD] KF %d saved: surf=%zu, corner=%zu, pos=[%.2f,%.2f,%.2f]\033[0m\n",
  //        kf_idx, kf_surf_ds->size(), kf_corner_ds->size(),
  //        kf_p[0], kf_p[1], kf_p[2]);

  // 分段写盘：interval > 0 且攒够了就落盘一段
  if (pcd_save_interval > 0 && kf_in_segment_ >= pcd_save_interval) {
    flushSegment(false);
  }
}
void LIVMapper::flushSegment(bool is_final)
{
  // 快照 + 清空（持锁完成，避免和发布线程/后续 finalizeKeyframe 打架）
  PointCloudXYZI::Ptr seg_traj (new PointCloudXYZI());
  PointCloudXYZI::Ptr seg_surf (new PointCloudXYZI());
  PointCloudXYZI::Ptr seg_corn (new PointCloudXYZI());

  int seg_idx_this;
  {
    std::lock_guard<std::mutex> lk(pcd_map_mtx_);

    if ((!global_trajectory_ || global_trajectory_->empty()) &&
        (!global_surf_map_   || global_surf_map_->empty())   &&
        (!global_corner_map_ || global_corner_map_->empty())) {
      return;  // 没东西可写
    }

    seg_idx_this = kf_segment_index_ + 1;  // 段号从 1 开始

    *seg_traj = *global_trajectory_;
    *seg_surf = *global_surf_map_;
    *seg_corn = *global_corner_map_;

    // 清空（trajectory 也清，保持 4 个文件一一对应；若想保留轨迹，改这里）
    global_trajectory_->clear();
    global_surf_map_->clear();
    global_corner_map_->clear();
    kf_in_segment_ = 0;
    kf_segment_index_ = seg_idx_this;
  }

  // 写盘
  const std::string suffix = "_" + std::to_string(seg_idx_this) + ".pcd";
  const std::string traj_path   = std::string(ROOT_DIR) + "Log/PCD/trajectory" + suffix;
  const std::string surf_path   = std::string(ROOT_DIR) + "Log/PCD/SurfMap"    + suffix;
  const std::string corner_path = std::string(ROOT_DIR) + "Log/PCD/CornerMap"  + suffix;
  const std::string global_path = std::string(ROOT_DIR) + "Log/PCD/GlobalMap"  + suffix;

  pcl::PCDWriter writer;

  if (!seg_traj->empty()) {
    writer.writeBinary(traj_path, *seg_traj);
    std::cout << GREEN << "[PCD][seg " << seg_idx_this << "] trajectory: "
              << seg_traj->size() << " kfs -> " << traj_path << RESET << std::endl;
  }
  if (!seg_surf->empty()) {
    writer.writeBinary(surf_path, *seg_surf);
    std::cout << GREEN << "[PCD][seg " << seg_idx_this << "] SurfMap: "
              << seg_surf->size() << " pts -> " << surf_path << RESET << std::endl;
  }
  if (!seg_corn->empty()) {
    writer.writeBinary(corner_path, *seg_corn);
    std::cout << GREEN << "[PCD][seg " << seg_idx_this << "] CornerMap: "
              << seg_corn->size() << " pts -> " << corner_path << RESET << std::endl;
  }

  // GlobalMap = Surf + Corner 合并再滤一次
  PointCloudXYZI::Ptr seg_global(new PointCloudXYZI());
  seg_global->reserve(seg_surf->size() + seg_corn->size());
  *seg_global += *seg_surf;
  *seg_global += *seg_corn;

  if (!seg_global->empty()) {
    PointCloudXYZI::Ptr seg_global_ds(new PointCloudXYZI());
    const float leaf = (filter_size_pcd > 0.01) ? (float)filter_size_pcd : 0.15f;
    voxelFilterManual(seg_global, seg_global_ds, leaf);
    writer.writeBinary(global_path, *seg_global_ds);
    std::cout << GREEN << "[PCD][seg " << seg_idx_this << "] GlobalMap: "
              << seg_global_ds->size() << " pts -> " << global_path << RESET << std::endl;
  }

  if (is_final) {
    std::cout << GREEN << "[PCD] Final segment " << seg_idx_this
              << " flushed on exit." << RESET << std::endl;
  }
}

void LIVMapper::cleanupLoopDetection() {
    if (std_manager_) {
        delete std_manager_;
        std_manager_ = nullptr;
    }
    kf_infos_.clear();
    frame_window_.clear();
}