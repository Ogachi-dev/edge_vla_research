cmake_minimum_required(VERSION 3.8)
project(evla_sim)
find_package(ament_cmake REQUIRED)

# 中身はファイルだけなので、share/evla_sim/ にコピーするだけ
install(DIRECTORY ___  # 【穴①】share に置くディレクトリの一覧
  DESTINATION share/${PROJECT_NAME})

ament_package()
