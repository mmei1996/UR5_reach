import gym
from gym import error,spaces,utils
from gym.utils import seeding
import math 
import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random
import time

class Ur5Env(gym.Env):
    max_steps_one_episode = 1000
    def __init__(self,is_render=False,is_good_view=False):

        self.is_render=is_render
        self.is_good_view=is_good_view

        if self.is_render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.x_low_obs=0.3
        self.x_high_obs=0.8
        self.y_low_obs=-0.3
        self.y_high_obs=0.3
        self.z_low_obs=0
        self.z_high_obs=0.55

        self.x_low_action=-1
        self.x_high_action=1
        self.y_low_action=-1
        self.y_high_action=1
        self.z_low_action=-1
        self.z_high_action=1

        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=20, cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])

        self.action_space=spaces.Box(low=np.array([self.x_low_action,self.y_low_action,self.z_low_action,-1,-1,-1]),
                                     high=np.array([self.x_high_action,self.y_high_action,self.z_high_action,1,1,1]),
                                     dtype=np.float32)
        self.observation_space=spaces.Box(low=np.array([self.x_low_obs,self.y_low_obs,self.z_low_obs]),
                                     high=np.array([self.x_high_obs,self.y_high_obs,self.z_high_obs]),
                                     dtype=np.float32)
        self.step_counter=0

        self.robot_jointVelocity = []

        self.urdf_root_path = pybullet_data.getDataPath()
        # lower limits for null space
        self.lower_limits = [-.7, -2, 0, -6, -6, -6]
        # upper limits for null space
        self.upper_limits = [.7, 0, 2.1, 6, 6, 6,]
        # joint ranges for null space
        self.joint_ranges = [1.4, 2, 2.1, 12, 12, 12, 12]
        # restposes for null space
        self.rest_poses = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66]
        # joint damping coefficents
        self.joint_damping = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]

        self.init_joint_positions = [-0.0061004455969415, -1.2992421540927381, 1.8041001952816262, -2.0756543764388575, -1.570796144056782 ,-0.006410070827403764]


        


        self.orientation = p.getQuaternionFromEuler([math.pi / 2, -math.pi, math.pi / 2.])

        self.target_x_param = p.addUserDebugParameter("Target X", -5, 5.0, 0.0)

        self.target_y_param = p.addUserDebugParameter("Target Y", -5, 5.0, 0.0)

        self.target_z_param = p.addUserDebugParameter("Target Z", 0.001, 0.1, 0.004)

        

        self.cri_distance = 0

        self.v = 0

        self.xx = 0

        self.yy = 0

        




        self.seed()
        self.reset()
    def range(self):
        # 危险圆
        radius1 = p.readUserDebugParameter(self.target_z_param)*100  # 圆的半径
        num_segments = 36  # 圆的线段数量，增加数量以增加圆的平滑度
        color = [1, 0, 0]  # 颜色（红色）
        center = [0, 0, 0]  # 圆心的位置

        # 计算圆上的点的坐标
        points = []
        for i in range(num_segments):
            angle = 2.0 * math.pi * i / num_segments
            x = radius1 * math.cos(angle) + center[0]
            y = radius1 * math.sin(angle) + center[1]
            points.append([x, y, center[2]])

        # 绘制圆的线段
        for i in range(num_segments):
            start_point = points[i]
            end_point = points[(i + 1) % num_segments]
            p.addUserDebugLine(start_point, end_point, color, lifeTime=0)

        # 安全圆
        radius = 2.3  # 圆的半径
        num_segments = 36  # 圆的线段数量，增加数量以增加圆的平滑度
        color = [0, 1, 0]  # 颜色（绿色）
        center = [0, 0, 0]  # 圆心的位置

        # 计算圆上的点的坐标
        points = []
        for i in range(num_segments):
            angle = 2.0 * math.pi * i / num_segments
            x = radius * math.cos(angle) + center[0]
            y = radius * math.sin(angle) + center[1]
            points.append([x, y, center[2]])

        # 绘制圆的线段
        for i in range(num_segments):
            start_point = points[i]
            end_point = points[(i + 1) % num_segments]
            p.addUserDebugLine(start_point, end_point, color, lifeTime=0)

            return radius1
        
    def reset(self):
        #p.connect(p.GUI)
        self.step_counter=0

        p.resetSimulation()
        #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.terminated=False
        p.setGravity(0, 0, 0)

        #这些是周围那些白线，用来观察是否超过了obs的边界
        p.addUserDebugLine(lineFromXYZ=[self.x_low_obs,self.y_low_obs,0],
                           lineToXYZ=[self.x_low_obs,self.y_low_obs,self.z_high_obs])
        p.addUserDebugLine(lineFromXYZ=[self.x_low_obs,self.y_high_obs,0],
                           lineToXYZ=[self.x_low_obs,self.y_high_obs,self.z_high_obs])
        p.addUserDebugLine(lineFromXYZ=[self.x_high_obs,self.y_low_obs,0],
                           lineToXYZ=[self.x_high_obs,self.y_low_obs,self.z_high_obs])
        p.addUserDebugLine(lineFromXYZ=[self.x_high_obs,self.y_high_obs,0],
                           lineToXYZ=[self.x_high_obs,self.y_high_obs,self.z_high_obs])

        p.addUserDebugLine(lineFromXYZ=[self.x_low_obs,self.y_low_obs,self.z_high_obs],
                           lineToXYZ=[self.x_high_obs,self.y_low_obs,self.z_high_obs])
        p.addUserDebugLine(lineFromXYZ=[self.x_low_obs,self.y_high_obs,self.z_high_obs],
                           lineToXYZ=[self.x_high_obs,self.y_high_obs,self.z_high_obs])
        p.addUserDebugLine(lineFromXYZ=[self.x_low_obs,self.y_low_obs,self.z_high_obs],
                           lineToXYZ=[self.x_low_obs,self.y_high_obs,self.z_high_obs])
        p.addUserDebugLine(lineFromXYZ=[self.x_high_obs,self.y_low_obs,self.z_high_obs],
                           lineToXYZ=[self.x_high_obs,self.y_high_obs,self.z_high_obs])
        


        p.loadURDF(os.path.join(self.urdf_root_path, "plane.urdf"), basePosition=[0, 0, -0.65])
        self.kuka_id = p.loadURDF("E:\\anaconda3\\Lib\\site-packages\\pybullet_data\\pybullet_ur5_robotiq-robotflow\\ur5.urdf",useFixedBase=True)
        p.loadURDF(os.path.join(self.urdf_root_path, "table/table.urdf"), basePosition=[0.1, 0, -0.65])
       # p.loadURDF(os.path.join(self.urdf_root_path, "tray/traybox.urdf"),basePosition=[0.55,0,0])
        #object_id=p.loadURDF(os.path.join(self.urdf_root_path, "random_urdfs/000/000.urdf"), basePosition=[0.53,0,0.02])
        self.object_id=p.loadURDF(os.path.join(self.urdf_root_path,"random_urdfs/000/000.urdf"),
                   basePosition=[random.uniform(self.x_low_obs,self.x_high_obs),
                                 random.uniform(self.y_low_obs,self.y_high_obs),
                                 0.3],useFixedBase=True)

        self.num_joints = p.getNumJoints(self.kuka_id)-1 #7


        x = random.uniform(-1.3, -0.5) if random.random() < 0.5 else random.uniform(0.5, 1.3)
        y = random.uniform(-1.3, -0.5) if random.random() < 0.5 else random.uniform(0.5, 1.3)

        self.humanrobot=p.loadURDF(os.path.join(self.urdf_root_path,"r2d2.urdf"), basePosition=[x,y,
                                 0],useFixedBase=True)

        
        

        for i in range(self.num_joints):
            p.resetJointState(bodyUniqueId=self.kuka_id,
                              jointIndex=i,
                              targetValue=self.init_joint_positions[i],
                              )

        self.robot_pos_obs=p.getLinkState(self.kuka_id,self.num_joints-1)[4] #末端橙色装置URDF文件中的坐标系相对于世界坐标系的位置
        #logging.debug("init_pos={}\n".format(p.getLinkState(self.kuka_id,self.num_joints-1)))
        p.stepSimulation()
        self.object_pos=p.getBasePositionAndOrientation(self.object_id)[0] #获取基座的坐标和方向
        #return np.array(self.object_pos).astype(np.float32)

        self.humanrobot_state=p.getLinkState(self.humanrobot,0)[0]
        # self.object_state=p.getBasePositionAndOrientation(self.object_id)
        # self.object_state=np.array(self.object_state).astype(np.float32)

        #


        self.robot_jointVelocity = []
        for i in range(self.num_joints):
            joint_state = p.getJointState(self.kuka_id, i)
            self.robot_jointVelocity.append(joint_state[0])

        self.robot_jointVelocity = tuple(self.robot_jointVelocity)

        #print(self.robot_pos_obs+self.robot_jointVelocity+self.humanrobot_state)


        

        return np.array(self.robot_pos_obs+self.robot_jointVelocity+self.humanrobot_state).astype(np.float32)
    
    def step(self,action):
        """
        dv=0.05
        dx=action[0]*dv
        dy=action[1]*dv
        dz=action[2]*dv"""
        dq=1
        q=[0 ,0, 0, 0, 0, 0]
        
        q[0]=action[0]*dq
        q[1]=action[1]*dq
        q[2]=action[2]*dq
        q[3]=action[3]*dq
        q[4]=action[4]*dq
        q[5]=action[5]*dq

        
        
        self.current_pos=p.getLinkState(self.kuka_id,self.num_joints-1)[4] #末端橙色装置URDF文件中的坐标系相对于世界坐标系的位置
        #logging.debug("init_pos={}\n".format(p.getLinkState(self.kuka_id,self.num_joints-1)))
       # logging.debug("self.current_pos={}\n".format(self.current_pos))
        """
        self.new_robot_pos=[self.current_pos[0]+dx,
                            self.current_pos[1]+dy,
                            self.current_pos[2]+dz]
        """
        #logging.debug("self.new_robot_pos={}\n".format(self.new_robot_pos))

        """
        self.robot_joint_positions = p.calculateInverseKinematics(
                                                       bodyUniqueId=self.kuka_id,
                                                       endEffectorLinkIndex=self.num_joints - 1, #末端橙色装置的索引
                                                       targetPosition=[self.new_robot_pos[0],
                                                                       self.new_robot_pos[1],
                                                                       self.new_robot_pos[2]],
                                                       targetOrientation=self.orientation,
                                                       jointDamping=self.joint_damping,
                                                       )
        """
        
        
        for i in range(self.num_joints):
            p.setJointMotorControl2(bodyUniqueId=self.kuka_id,
                                    jointIndex=i,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=q[i]
                                    )
        
        p.stepSimulation()

        #在代码开始部分，如果定义了is_good_view，那么机械臂的动作会变慢，方便观察
        if self.is_good_view:
            time.sleep(0.05)

        self.step_counter += 1
        
        
        return self._reward()
    
    def human(self):
        slideId1_value = p.readUserDebugParameter(self.target_x_param)
        slideId2_value = p.readUserDebugParameter(self.target_y_param)
        p.resetBasePositionAndOrientation(self.humanrobot, [slideId1_value+2, slideId2_value+2, -0.2], [0, 0, 0, 1])


        #print("cri_distance:",self.cri_distance,"velocity:",self.v)
        p.stepSimulation()
        
        #在代码开始部分，如果定义了is_good_view，那么机械臂的动作会变慢，方便观察
        return slideId1_value
        

    def goal_state(self):
        self.object_state=np.array(
            p.getBasePositionAndOrientation(self.object_id)[0]
        ).astype(np.float32)

        return np.array(self.object_state).astype(np.float32)

    def _reward(self):

        #一定注意是取第4个值，请参考pybullet手册的这个函数返回值的说明
        self.robot_state=p.getLinkState(self.kuka_id,self.num_joints-1)[4]

        self.humanrobot_state=p.getLinkState(self.humanrobot,0)[0]
        # self.object_state=p.getBasePositionAndOrientation(self.object_id)
        # self.object_state=np.array(self.object_state).astype(np.float32)

        #
        xx = self.humanrobot_state[0]
        yy = self.humanrobot_state[1]
        zz = self.humanrobot_state[2]
        cri_dis = math.sqrt(xx**2+yy**2+zz**2)

        self.robot_jointVelocity = []
        for i in range(self.num_joints):
            joint_state = p.getJointState(self.kuka_id, i)
            self.robot_jointVelocity.append(joint_state[1])

        self.robot_jointVelocity = tuple(self.robot_jointVelocity)
            

        self.object_state=np.array(
            p.getBasePositionAndOrientation(self.object_id)[0]
        ).astype(np.float32)

        square_dx=(self.robot_state[0]-self.object_state[0])**2
        square_dy=(self.robot_state[1]-self.object_state[1])**2
        square_dz = (self.robot_state[2] - self.object_state[2]) ** 2
        dz = self.robot_state[2] - self.object_state[2]

        #用机械臂末端和物体的距离作为奖励函数的依据
        self.distance=math.sqrt(square_dx+square_dy+square_dz)
        #print(cri_dis)

        x=self.robot_state[0]
        y=self.robot_state[1]
        z=self.robot_state[2]

        #如果机械比末端超过了obs的空间，也视为done，而且会给予一定的惩罚
        terminated=bool(
            x<self.x_low_obs
            or x>self.x_high_obs
            or y<self.y_low_obs
            or y>self.y_high_obs
            or z<self.z_low_obs
            or z>self.z_high_obs
        )

        if terminated:
            reward = -0.1
            self.terminated = True

        #如果机械臂一直无所事事，在最大步数还不能接触到物体，也需要给一定的惩罚
        elif self.step_counter > self.max_steps_one_episode and cri_dis>1.5:
            reward = -0.1
            self.terminated = True


        elif self.distance < 0.1:
            reward = 1
            self.terminated = True

            
        else:
            reward = 0
            self.terminated = False

        info={'distance:',self.distance}
        self.observation=self.robot_state + self.robot_jointVelocity + self.humanrobot_state
        #self.observation=self.object_state








        #print(self.observation)

        return np.array(self.observation).astype(np.float32),reward,self.terminated,info

    def render(self):
        view_matrix=p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7,0,0.05],
                                                        distance=.7,
                                                        yaw=90,
                                                        pitch=-70,
                                                        roll=0,upAxisIndex=2)
        proj_matrix=p.computeProjectionMatrixFOV(fov=60,aspect=float(960)/720,  #投影矩阵是一个4x4的矩阵，它用于将场景的三维坐标转换为二维屏幕空间中的坐标。
                                                 nearVal=0.1,
                                                 farVal=100.0)
        (_,_,px,_,_)=p.getCameraImage(width=960,height=720,
                                      viewMatrix=view_matrix,
                                      projectionMatrix=proj_matrix,
                                      renderer=p.ER_BULLET_HARDWARE_OPENGL) #list of [char RED,char GREEN,char BLUE,char ALPHA] [0..width*height]
        rgb_array=np.array(px,dtype=np.uint8) #转化为np数组，数据类型为无符号8位整型
        rgb_array=np.reshape(rgb_array,(720,960,4))
        rgb_array=rgb_array[:,:,:3] #取前3个通道
        return rgb_array

if __name__ == '__main__':
    env=Ur5Env(is_render=True,is_good_view=True)
    env.reset()
    obs = env.reset()
    for i in range(100):
        env.reset()
    
        for i in range(20):
            action=env.action_space.sample()
            
            obs,reward,done,info=env.step(action)
        
            
           
            if done:
                break
           # time.sleep(0.1)
        
        
    env.close()
    