#ifndef GAUSSIAN_SPARSE_SLAM_SERIALIZER_H
#define GAUSSIAN_SPARSE_SLAM_SERIALIZER_H
#pragma once

#include <vector>
#include <string>
#include <cstring>
#include <cstdint>

struct LocalKeyframeData {
    std::string user;
    int16_t keyframe_id;
    std::vector<int16_t> indices;
    std::vector<int16_t> mp_ids;
};
namespace GaussianSparseSLAM {
    class KeyframeSerializer {
    public:
        /**
         * 키프레임 데이터를 바이트 배열로 직렬화
         * @return: 바이트 데이터 (std::vector<uint8_t>)
         */
        static void serialize(std::vector<uint8_t>& buffer, const std::vector<LocalKeyframeData>& keyframes) {
            
            // 예상 크기 미리 할당 (성능 최적화) 
            size_t estimated_size = 1;  // 헤더. 로컬 맵의 키프레임 수.
            //이름 크기(1), 인덱스 크기(2), 스트링, 아이디(2), 인덱스*2
            for (const auto& kf : keyframes) {
                estimated_size += 1 + 2 + kf.user.length() + 2 + kf.indices.size() * 2;
            }
            buffer.reserve(estimated_size);

            // 1. 헤더: 총 키프레임 개수 (4 bytes)
            uint8_t num_keyframes = static_cast<uint8_t>(keyframes.size());
            appendBytes(buffer, num_keyframes);

            // 2. 각 키프레임 데이터 직렬화
            for (const auto& kf : keyframes) {
                serializeKeyframe(buffer, kf);
            }
        }

        // 바이트 배열의 데이터 포인터와 크기 반환
        struct ByteView {
            const uint8_t* data;
            size_t size;
        };

        static ByteView getByteView(const std::vector<uint8_t>& buffer) {
            return { buffer.data(), buffer.size() };
        }

    private:
        static void serializeKeyframe(std::vector<uint8_t>& buffer,
            const LocalKeyframeData& kf) {
            // 1. 사용자 이름 길이 (1 byte)
            uint8_t name_len = static_cast<uint8_t>(kf.user.length());
            buffer.push_back(name_len);

            // 2. 인덱스 개수 (2 bytes)
            int16_t num_indices = static_cast<int16_t>(kf.indices.size());
            appendBytes(buffer, num_indices);

            // 3. 사용자 이름 (N bytes)
            buffer.insert(buffer.end(), kf.user.begin(), kf.user.end());

            // 4. 키프레임 ID (2 bytes)
            appendBytes(buffer, kf.keyframe_id);

            // 5. 인덱스 배열 (N * 2 bytes)
            for (int16_t idx : kf.indices) {
                appendBytes(buffer, idx);
            }
            //std::cout << kf.user << "," << kf.keyframe_id << "," << num_indices <<" = "<<kf.mp_ids.size() << std::endl;
        }

        template<typename T>
        static void appendBytes(std::vector<uint8_t>& buffer, T value) {
            const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
            buffer.insert(buffer.end(), bytes, bytes + sizeof(T));
        }
    };

}

#endif