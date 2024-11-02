# 灯条识别与匹配
## 基本思路
1. 降低亮度
2. 将图片只保留红色部分
3. 转换成灰度图
4. 边缘增强
5. 二值化
6. 膨胀，腐蚀
7. 检测并保存边缘信息
8. 轮廓近似
9. 通过长宽比判断轮廓是否为灯管状
10. 检查轮廓是否平行并光滑
11. 将平行轮廓放入容器
12. 将存有平行轮廓的容器存入新的容器
13. 绘制轮廓，显示图像
[流程图](https://github.com/ljk4/3rd_kaohe/edit/main/)
