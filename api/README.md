## 启动测试沙箱容器命令
docker run -d -p 8080:8080 -p 5900:5900 -p 5901:5901 -p 9222:9222 --name sandbox-dev sandbox-dev


## 测试环境沙箱启动命令

```bash
# 1.构建API镜像
docker build -t manus-api-dev .

# 2.创建docker网络
docker network create manus-network-dev

# 3.启动redis开发容器
docker run -d --name manus-redis-dev --network manus-network-dev -p 6379:6379 -v manus_redis_data_dev:/data redis:8.2

# 4.启动postgres开发容器
docker run -d --name manus-db-dev --network manus-network-dev -p 5432:5432 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=manus -v manus_postgres_data_dev:/var/lib/postgresql/data postgres:17.6

# 5.运行api开发容器
docker run -d --name manus-api-dev --network manus-network-dev -p 8000:8000 -v /var/run/docker.sock:/var/run/docker.sock:ro manus-api-dev
```