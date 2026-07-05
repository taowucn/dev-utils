import pcl.pcl_visualization


# lidar_path 指定一个kitti 数据的点云bin文件就行了
points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)  # .astype(np.float16)
cloud = pcl.PointCloud(points[:,:3])
visual = pcl.pcl_visualization.CloudViewing()
visual.ShowMonochromeCloud(cloud)


flag = True
while flag:
    flag != visual.WasStopped()

