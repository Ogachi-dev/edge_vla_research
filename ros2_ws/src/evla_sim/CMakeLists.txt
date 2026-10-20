cmake_minimum_required(VERSION 3.8)
project(evla_sim)
find_package(ament_cmake REQUIRED)

# 中身はファイルだけなので、share/evla_sim/ にコピーするだけ
install(DIRECTORY urdf config worlds launch scripts  # 【穴①】share に置くディレクトリの一覧
  DESTINATION share/${PROJECT_NAME})

ament_package()
