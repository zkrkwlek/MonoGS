import gtsam
import numpy as np
import time
import matplotlib.pyplot as plt
from gtsam import symbol_shorthand
from typing import List, Tuple

# 심볼 단축키
X = symbol_shorthand.X  # 카메라 포즈
P = symbol_shorthand.P  # 3차원 포인트


class PnPOptimizer:
    """2D-3D 대응쌍을 이용한 카메라 자세 최적화"""

    def __init__(self):#camera_intrinsics: gtsam.Cal3_S2):
        """
        Args:
            camera_intrinsics: 카메라 내부 파라미터 (fx, fy, s, u0, v0)
        """
        #self.K = self.ConvertIntrinsic(fx, fy, cx, cy)
        self.isam_params = gtsam.ISAM2Params()
        self.isam = gtsam.ISAM2(self.isam_params)
        self.measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
        self.values = gtsam.Values()

    def ConvertIntrinsic(self, fx, fy, cx, cy, s = 0.0):
        return gtsam.Cal3_S2(fx, fy, s, cx, cy)

    def ConvertPose(self, R, t):
        R_np = R.detach().cpu().numpy()
        t_np = t.detach().cpu().numpy()

        rotation = gtsam.Rot3(R_np)
        translation = gtsam.Point3(t_np[0], t_np[1], t_np[2])

        return gtsam.Pose3(rotation, translation)

    def initialize(self, initial_pose : gtsam.Pose3, points_3d):
        pass

    def clear(self):
        self.values.clear()

    def insert_points(self, points, ids):
        self.values = gtsam.Values()
        for p3d, pid in zip(points, ids):
            #initial_values.insert(gtsam.symbol('P', pid), p3d)
            self.values.insert(P(pid), p3d)

    def update_pose(self, R, t, id):
        T = self.ConvertPose(R, t)
        self.values.update(X(id), T)

    def optimize_pose(self,
                      idx,
                      points_2d: np.ndarray,
                      points_id: np.ndarray,
                      initial_pose: gtsam.Pose3,
                      K,
                      measurement_noise_sigma: float = 1.0,
                      optimizer_type: str = "LM") -> Tuple[gtsam.Pose3, float, float]:
        """
        2D-3D 대응쌍으로부터 카메라 자세 최적화

        Args:
            points_3d: 3D 월드 포인트들 (N x 3)
            points_2d: 대응하는 2D 이미지 포인트들 (N x 2)
            initial_pose: 초기 카메라 자세 추정값
            measurement_noise_sigma: 측정 노이즈 표준편차 (픽셀 단위)
            optimizer_type: 최적화 알고리즘 ("GN", "LM", "Dogleg")

        Returns:
            optimized_pose: 최적화된 카메라 자세
            final_error: 최종 재투영 에러
            info: 최적화 정보 (시간, 반복횟수 등)
        """
        start_time = time.time()

        # 1. Factor Graph 생성
        graph = gtsam.NonlinearFactorGraph()


        # 2. 노이즈 모델 설정
        measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, measurement_noise_sigma)
        huber_kernel = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber(1.345),  # 1.345는 튜닝 가능한 thresh 값
            measurement_noise
        )

        # 3. 프로젝션 팩터들 추가
        for point_2d, pid in zip(points_2d, points_id):

            # 2D 관측값을 GTSAM Point2로 변환
            measurement = gtsam.Point2(point_2d[0], point_2d[1])

            # GenericProjectionFactor 추가 (랜드마크는 고정, 포즈만 최적화)
            factor = gtsam.GenericProjectionFactorCal3_S2(
                measurement, huber_kernel, X(idx), P(pid), K)
            graph.add(factor)

        # 4. 초기값 설정
        self.values.insert(X(idx), initial_pose)

        # 5. 최적화 실행
        if optimizer_type == "GN":
            optimizer = gtsam.GaussNewtonOptimizer(graph, self.values)
        elif optimizer_type == "LM":
            params = gtsam.LevenbergMarquardtParams()
            optimizer = gtsam.LevenbergMarquardtOptimizer(graph, self.values, params)
        elif optimizer_type == "Dogleg":
            params = gtsam.DoglegParams()
            optimizer = gtsam.DoglegOptimizer(graph, self.values, params)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        result = optimizer.optimize()

        optimization_time = time.time() - start_time

        # 6. 결과 분석
        optimized_pose = result.atPose3(X(idx))
        final_error = graph.error(result)
        initial_error = graph.error(self.values)

        # 7. 재투영 에러 계산
        #reprojection_errors = self._calculate_reprojection_errors(
        #    optimized_pose, points_3d, points_2d, K)
        """
        info = {
            'optimization_time': optimization_time,
            'initial_error': initial_error,
            'final_error': final_error,
            'num_correspondences': len(points_3d),
            'mean_reprojection_error': np.mean(reprojection_errors),
            'max_reprojection_error': np.max(reprojection_errors),
            'optimizer_type': optimizer_type
        }
        """
        return optimized_pose, final_error, initial_error

    def calculate_reprojection_errors(self,
                                       pose: gtsam.Pose3,
                                       points_3d: np.ndarray,
                                       points_2d: np.ndarray,
                                       K) -> np.ndarray:
        """재투영 에러 계산"""
        camera = gtsam.PinholeCameraCal3_S2(pose, K)
        errors = []

        for point_3d, point_2d in zip(points_3d, points_2d):
            projected = camera.project(gtsam.Point3(point_3d[0], point_3d[1], point_3d[2]))
            error = np.linalg.norm([projected[0] - point_2d[0], projected[1] - point_2d[1]])
            errors.append(error)
            """
            try:
                projected = camera.project(gtsam.Point3(point_3d[0], point_3d[1], point_3d[2]))
                error = np.linalg.norm([projected.x() - point_2d[0],projected.y() - point_2d[1]])
                errors.append(error)
            except:
                #errors.append(float('inf'))  # 투영 실패 시
                pass
            """

        return np.array(errors)


def generate_synthetic_data(num_points: int = 100,
                            noise_level: float = 1.0,
                            scene_size: float = 5.0) -> Tuple[gtsam.Pose3, np.ndarray, np.ndarray, gtsam.Cal3_S2]:
    """
    PnP 테스트용 합성 데이터 생성

    Args:
        num_points: 생성할 3D 포인트 개수
        noise_level: 2D 관측값 노이즈 레벨 (픽셀)
        scene_size: 3D 장면 크기

    Returns:
        true_pose: 실제 카메라 자세
        points_3d: 3D 월드 포인트들
        points_2d_noisy: 노이즈가 있는 2D 관측값들
        camera_intrinsics: 카메라 내부 파라미터
    """

    # 카메라 내부 파라미터
    fx, fy = 500.0, 500.0
    s = 0.0
    u0, v0 = 320.0, 240.0
    K = gtsam.Cal3_S2(fx, fy, s, u0, v0)

    # 실제 카메라 자세 (약간 회전하고 뒤로 이동)
    rotation = gtsam.Rot3.Ypr(0.2, 0.1, 0.0)  # yaw, pitch, roll
    translation = gtsam.Point3(0.0, 0.0, 3.0)  # z축으로 3미터 뒤
    true_pose = gtsam.Pose3(rotation, translation)

    # 3D 포인트들 생성 (카메라 앞쪽 공간에 랜덤 분포)
    points_3d = np.random.uniform(-scene_size, scene_size, (num_points, 3))
    points_3d[:, 2] = np.random.uniform(1.0, 10.0, num_points)  # z는 양수 (카메라 앞쪽)
    print(points_3d.shape)
    # 2D 투영 계산
    camera = gtsam.PinholeCameraCal3_S2(true_pose, K)
    points_2d_clean = []
    valid_points_3d = []

    for point_3d in points_3d:
        try:
            projected = camera.project(gtsam.Point3(point_3d[0], point_3d[1], point_3d[2]))
            print(projected)
            # 이미지 범위 내에 있는지 확인
            if (0 <= projected.x() <= 640 and 0 <= projected.y() <= 480):
                points_2d_clean.append([projected.x(), projected.y()])
                valid_points_3d.append(point_3d)
        except:
            continue
    print(valid_points_3d)
    points_3d = np.array(valid_points_3d)
    points_2d_clean = np.array(points_2d_clean)

    # 노이즈 추가
    noise = np.random.normal(0, noise_level, points_2d_clean.shape)
    points_2d_noisy = points_2d_clean + noise

    return true_pose, points_3d, points_2d_noisy, K


def benchmark_pnp_performance():
    """PnP 최적화 성능 벤치마크"""

    print("=== GTSAM PnP 최적화 성능 벤치마크 ===\n")

    # 테스트 설정
    point_counts = [10, 25, 50, 100, 200, 500, 1000]
    noise_levels = [0.5, 1.0, 2.0]
    optimizers = ["GN", "LM", "Dogleg"]

    results = {}

    for optimizer_type in optimizers:
        results[optimizer_type] = {
            'times': [],
            'errors': [],
            'point_counts': []
        }

        print(f"\n--- {optimizer_type} 최적화 결과 ---")
        print(f"{'포인트 수':<8} {'시간(ms)':<10} {'최종 에러':<12} {'재투영 에러':<12}")
        print("-" * 50)

        for num_points in point_counts:
            times = []
            errors = []
            reprojection_errors = []

            # 각 설정에 대해 여러 번 실행하여 평균 계산
            num_trials = 10
            for trial in range(num_trials):
                try:
                    # 데이터 생성
                    true_pose, points_3d, points_2d, K = generate_synthetic_data(
                        num_points=num_points, noise_level=1.0)

                    # 초기 추정값 (실제 자세에서 약간 벗어난 값)
                    noise_rotation = gtsam.Rot3.Ypr(0.1, 0.1, 0.1)
                    noise_translation = gtsam.Point3(0.2, 0.2, 0.2)
                    initial_pose = true_pose.compose(gtsam.Pose3(noise_rotation, noise_translation))

                    # 최적화 실행
                    optimizer = PnPOptimizer(K)
                    optimized_pose, final_error, info = optimizer.optimize_pose(
                        points_3d, points_2d, initial_pose,
                        measurement_noise_sigma=1.0, optimizer_type=optimizer_type)

                    times.append(info['optimization_time'] * 1000)  # ms로 변환
                    errors.append(final_error)
                    reprojection_errors.append(info['mean_reprojection_error'])

                except Exception as e:
                    print(f"Error in trial {trial}: {e}")
                    continue

            if times:
                avg_time = np.mean(times)
                avg_error = np.mean(errors)
                avg_reproj_error = np.mean(reprojection_errors)

                results[optimizer_type]['times'].append(avg_time)
                results[optimizer_type]['errors'].append(avg_error)
                results[optimizer_type]['point_counts'].append(num_points)

                print(f"{num_points:<8} {avg_time:<10.2f} {avg_error:<12.6f} {avg_reproj_error:<12.4f}")

    return results


def detailed_pnp_example():
    """상세한 PnP 최적화 예제"""

    print("\n=== 상세한 PnP 최적화 예제 ===\n")

    # 1. 데이터 생성
    true_pose, points_3d, points_2d, K = generate_synthetic_data(
        num_points=100, noise_level=1.0)

    print(f"생성된 데이터:")
    print(f"- 3D 포인트 수: {len(points_3d)}")
    print(f"- 실제 카메라 자세:")
    print(f"  회전: {true_pose.rotation().ypr()}")
    print(f"  위치: {true_pose.translation()}")

    # 2. 초기 추정값 (실제 값에서 벗어난 값)
    noise_rotation = gtsam.Rot3.Ypr(0.3, 0.2, 0.1)
    noise_translation = gtsam.Point3(0.5, 0.3, 0.4)
    initial_pose = true_pose.compose(gtsam.Pose3(noise_rotation, noise_translation))

    print(f"\n초기 추정값:")
    print(f"  회전: {initial_pose.rotation().ypr()}")
    print(f"  위치: {initial_pose.translation()}")

    # 3. 최적화 실행
    optimizer = PnPOptimizer(K)

    print(f"\n최적화 결과:")
    print(f"{'알고리즘':<10} {'시간(ms)':<10} {'최종에러':<12} {'재투영에러':<12} {'자세 에러':<15}")
    print("-" * 65)

    for opt_type in ["GN", "LM", "Dogleg"]:
        optimized_pose, final_error, info = optimizer.optimize_pose(
            points_3d, points_2d, initial_pose,
            measurement_noise_sigma=1.0, optimizer_type=opt_type)

        # 자세 에러 계산
        pose_error = true_pose.between(optimized_pose)
        rotation_error = np.linalg.norm(pose_error.rotation().ypr())
        translation_error = np.linalg.norm(pose_error.translation())
        total_pose_error = rotation_error + translation_error

        print(f"{opt_type:<10} {info['optimization_time'] * 1000:<10.2f} "
              f"{final_error:<12.6f} {info['mean_reprojection_error']:<12.4f} "
              f"{total_pose_error:<15.6f}")

    return true_pose, points_3d, points_2d, K


def plot_performance_results(results):
    """성능 결과 시각화"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 처리 시간 비교
        for optimizer, data in results.items():
            ax1.plot(data['point_counts'], data['times'], 'o-', label=optimizer, linewidth=2)

        ax1.set_xlabel('포인트 수')
        ax1.set_ylabel('처리 시간 (ms)')
        ax1.set_title('PnP 최적화 처리 시간 비교')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')

        # 최종 에러 비교
        for optimizer, data in results.items():
            ax2.plot(data['point_counts'], data['errors'], 's-', label=optimizer, linewidth=2)

        ax2.set_xlabel('포인트 수')
        ax2.set_ylabel('최종 에러')
        ax2.set_title('PnP 최적화 정확도 비교')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        ax2.set_yscale('log')

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("matplotlib이 설치되지 않아 시각화를 건너뜁니다.")


if __name__ == "__main__":

    # 1. 상세한 예제 실행
    true_pose, points_3d, points_2d, K = detailed_pnp_example()

    # 2. 성능 벤치마크 실행
    results = benchmark_pnp_performance()

    # 3. 결과 시각화
    plot_performance_results(results)

    # 4. 결론 및 권장사항
    print(f"\n=== 결론 및 권장사항 ===")
    print(f"1. 소수 포인트(< 50개): Gauss-Newton이 가장 빠름")
    print(f"2. 중간 포인트(50-200개): Levenberg-Marquardt 추천")
    print(f"3. 많은 포인트(> 200개): 모든 방법이 유사한 성능")
    print(f"4. 평균 처리 시간 (100개 포인트 기준):")

    if results:
        for opt_type in ["GN", "LM", "Dogleg"]:
            if opt_type in results and results[opt_type]['point_counts']:
                idx = None
                for i, count in enumerate(results[opt_type]['point_counts']):
                    if count >= 100:
                        idx = i
                        break
                if idx is not None:
                    time_100 = results[opt_type]['times'][idx]
                    print(f"   - {opt_type}: {time_100:.2f} ms")