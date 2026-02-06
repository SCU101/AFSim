import math
import numpy as np


# 数学计算工具
class RAMathUtil:
    # 弧度转度
    @staticmethod
    def Rad2Deg(AngRad):
        return AngRad * 57.2957795131

    # 度转弧度
    @staticmethod
    def Deg2Rad(AngDeg):
        return AngDeg * 0.01745329252

    @staticmethod
    def convert_lat_long_to_xy(pos_lat_lon, center):
        """
        将经纬度坐标转换为平面笛卡尔坐标
        :param pos_latlong: 字典，包含 'lat' 和 'lon' 键，表示待转换点的经纬度（度数）
        :param center: 字典，包含 'lat' 和 'lon' 键，表示参考中心点的经纬度（度数）
        :return: 字典，包含 'x' 和 'y' 键，表示转换后的平面坐标（米）
        """
        r = 6371000  # 地球半径（米）

        # 将度数转换为弧度
        pos_lat = RAMathUtil.Deg2Rad(pos_lat_lon['lat'])
        pos_lon = RAMathUtil.Deg2Rad(pos_lat_lon['lon'])
        center_lat = RAMathUtil.Deg2Rad(center['lat'])
        center_lon = RAMathUtil.Deg2Rad(center['lon'])

        delta_lon = pos_lon - center_lon

        # 计算临时变量（避免重复计算）
        tmp = (math.sin(pos_lat) * math.sin(center_lat) +
               math.cos(pos_lat) * math.cos(center_lat) * math.cos(delta_lon))

        # 计算平面坐标
        x = (r * math.cos(pos_lat) * math.sin(delta_lon)) / tmp
        y = (r * (math.sin(pos_lat) * math.cos(center_lat) -
                  math.cos(pos_lat) * math.sin(center_lat) * math.cos(delta_lon))) / tmp
        return x, y

    @staticmethod
    def convert_xy_to_lat_long(center_lat_lon, delta_x, delta_y, delta_z=0):
        """
        将相对平面坐标转换为经纬度坐标

        参数:
            center_lat_lon: 字典，参考中心点的经纬度 {'lat': xx, 'lon': yy} (度数)
            delta_x: 相对于中心的东向偏移 (米, 东为正)
            delta_y: 相对于中心的北向偏移 (米, 北为正)
            delta_z: 高度偏移 (米, 可选，默认0)

        返回:
            字典，包含 'lat', 'lon', 'alt' 的目标点坐标
        """
        # 地球半径 (米)
        r = 6371000

        # 将中心点经纬度转换为弧度
        lat0 = math.radians(center_lat_lon['lat'])
        lon0 = math.radians(center_lat_lon['lon'])

        # 初始高度 (如果有的话)
        alt0 = center_lat_lon.get('alt', 0)

        # 计算角距离 (沿着大圆的距离)
        d = math.sqrt(delta_x ** 2 + delta_y ** 2)

        if d == 0:
            # 如果没有平面位移，直接返回中心点
            return {
                'lat': center_lat_lon['lat'],
                'lon': center_lat_lon['lon'],
                'alt': alt0 + delta_z
            }

        # 计算方位角 (从北方向顺时针)
        azimuth = math.atan2(delta_x, delta_y)  # 注意：atan2(delta_x, delta_y) 对应北东坐标系

        # 计算角距离对应的圆心角
        angular_distance = d / r

        # 计算目标点的纬度
        lat = math.asin(
            math.sin(lat0) * math.cos(angular_distance) +
            math.cos(lat0) * math.sin(angular_distance) * math.cos(azimuth)
        )

        # 计算目标点的经度
        lon = lon0 + math.atan2(
            math.sin(azimuth) * math.sin(angular_distance) * math.cos(lat0),
            math.cos(angular_distance) - math.sin(lat0) * math.sin(lat)
        )

        # 将弧度转换回度数
        lat_deg = math.degrees(lat)
        lon_deg = math.degrees(lon)

        # 规范化经度到 [-180, 180] 范围
        lon_deg = (lon_deg + 180) % 360 - 180

        return {
            'lat': lat_deg,
            'lon': lon_deg,
            'alt': alt0 + delta_z
        }

    @staticmethod
    def generate_target_arc(current_pos=None, min_dist=12000, max_dist=15000):
        """
        简洁版：在12-15km圆弧内生成随机目标点

        参数:
            current_pos: 当前位置 [x, y, z]，默认[0,0,0]
            min_dist: 最小距离，默认12000米（12km）
            max_dist: 最大距离，默认15000米（15km）

        返回:
            target_pos: 目标点坐标 [x, y, z]
        """
        # 默认当前位置
        if current_pos is None:
            current_pos = np.array([0.0, 0.0, 0.0])

        # 随机角度 (0到2π)
        angle = np.random.uniform(0, 2 * math.pi)

        # 随机距离 (12-15km)
        distance = np.random.uniform(min_dist, max_dist)

        # 计算目标点
        target_x = current_pos[0] + distance * math.cos(angle)
        target_y = current_pos[1] + distance * math.sin(angle)
        target_z = 0

        return np.array([target_x, target_y, target_z])


# 定义 一个 TSVector3D
class BaseTSVector3:
    # 初始化
    def __init__(self, x: float, y: float, z: float):
        return {"X": x, "Y": y, "Z": z}

    # 矢量a + 矢量b
    @staticmethod
    def plus(a, b):
        return {"X": a["X"] + b["X"], "Y": a["Y"] + b["Y"], "Z": a["Z"] + b["Z"]}

    # 矢量a - 矢量b
    @staticmethod
    def minus(a, b):
        return {"X": a["X"] - b["X"], "Y": a["Y"] - b["Y"], "Z": a["Z"] - b["Z"]}

    # 矢量a * 标量scal
    @staticmethod
    def multscalar(a, scal):
        return {"X": a["X"] * scal, "Y": a["Y"] * scal, "Z": a["Z"] * scal}

    # 矢量a / 标量scal
    @staticmethod
    def divdbyscalar(a, scal):
        if scal == 0:
            return {"X": 1.633123935319537e+16, "Y": 1.633123935319537e+16, "Z": 1.633123935319537e+16}
        else:
            return {"X": a["X"] / scal, "Y": a["Y"] / scal, "Z": a["Z"] / scal}

    # 矢量a 点乘 矢量b
    @staticmethod
    def dot(a, b):
        return a["X"] * b["X"] + a["Y"] * b["Y"] + a["Z"] * b["Z"]

    # 矢量a 叉乘 矢量b
    @staticmethod
    def cross(a, b):
        val = {"X": a["Y"] * b["Z"] - a["Z"] * b["Y"], \
               "Y": a["Z"] * b["X"] - a["X"] * b["Z"], \
               "Z": a["X"] * b["Y"] - a["Y"] * b["X"]}
        return val

    # 判断矢量a是否为0矢量
    @staticmethod
    def iszero(a):
        if a["X"] == 0 and a["Y"] == 0 and a["Z"] == 0:
            return True
        else:
            return False

    # 矢量a归一化
    @staticmethod
    def normalize(a):
        vallen = math.sqrt(a["X"] * a["X"] + a["Y"] * a["Y"] + a["Z"] * a["Z"])
        val = {"X": 0, "Y": 0, "Z": 0}
        if vallen > 0:
            val = {"X": a["X"] / vallen, "Y": a["Y"] / vallen, "Z": a["Z"] / vallen}
        return val

    # 计算矢量a的长度
    @staticmethod
    def length(a):
        if a["X"] == 0 and a["Y"] == 0 and a["Z"] == 0:
            return 0
        return math.sqrt(a["X"] * a["X"] + a["Y"] * a["Y"] + a["Z"] * a["Z"])

    # 计算矢量a的长度平方
    @staticmethod
    def lengthsqr(a):
        return a["X"] * a["X"] + a["Y"] * a["Y"] + a["Z"] * a["Z"]


# 三维矢量计算工具
class TSVector3(BaseTSVector3):
    # 初始化
    def __init__(self, x: float, y: float, z: float):
        return {"X": x, "Y": y, "Z": z}

    # 计算位置矢量a与位置矢量b间的距离
    @staticmethod
    def distance(a, b):
        return BaseTSVector3.length(BaseTSVector3.minus(a, b))

    # 计算位置矢量a与位置矢量b间的距离平方
    @staticmethod
    def distancesqr(a, b):
        return BaseTSVector3.lengthsqr(BaseTSVector3.minus(a, b))

    # 计算矢量a与矢量b之间的夹角，单位弧度
    @staticmethod
    def angle(a, b):
        if BaseTSVector3.iszero(a) or BaseTSVector3.iszero(b):
            return 0
        else:
            ma = BaseTSVector3.length(a)
            mb = BaseTSVector3.length(b)
            mab = BaseTSVector3.dot(a, b)
        return math.acos(mab / ma / mb)

    # 给定方位角heading和俯仰角pitch，单位弧度，计算单位方向矢量
    @staticmethod
    def calorientation(heading, pitch):
        return {"X": math.sin(heading) * math.cos(pitch), "Y": math.cos(heading) * math.cos(pitch),
                "Z": math.sin(pitch)}

    # 计算矢量direction的方位角，单位弧度
    @staticmethod
    def calheading(direction):
        if BaseTSVector3.iszero(direction):
            return 0
        else:
            heading = math.atan2(direction["X"], direction["Y"])
            if heading < 0:
                heading += math.pi * 2
            return heading

    # 计算矢量direction的方位角，单位度
    @staticmethod
    def calheading_deg(direction):
        if BaseTSVector3.iszero(direction):
            return 0
        else:
            heading = math.atan2(direction["X"], direction["Y"])
            if heading < 0:
                heading += math.pi * 2
            return RAMathUtil.Rad2Deg(heading)

    # 计算矢量direction的俯仰角，单位弧度
    @staticmethod
    def calpitch(direction):
        if BaseTSVector3.iszero(direction):
            return 0
        elif direction["X"] == 0 and direction["Y"] == 0:
            return math.pi * 0.5
        elif direction["Z"] == 0:
            return 0
        else:
            mxy = math.sqrt(direction["X"] * direction["X"] + direction["Y"] * direction["Y"])
            return math.atan2(direction["Z"], mxy)

    # 计算矢量direction的俯仰角，单位度
    @staticmethod
    def calpitch_deg(direction):
        if BaseTSVector3.iszero(direction):
            return 0
        elif direction["X"] == 0 and direction["Y"] == 0:
            return math.pi * 0.5
        elif direction["Z"] == 0:
            return 0
        else:
            mxy = math.sqrt(direction["X"] * direction["X"] + direction["Y"] * direction["Y"])
            return RAMathUtil.Rad2Deg(math.atan2(direction["Z"], mxy))

    # 计算位置矢量pos1与位置矢量pos2之间的地面距离
    @staticmethod
    def groundrange(pos1, pos2):
        return math.sqrt((pos1["X"] - pos1["X"]) * (pos1["X"] - pos1["X"]) + \
                         (pos1["Y"] - pos1["Y"]) * (pos1["Y"] - pos1["Y"]))

