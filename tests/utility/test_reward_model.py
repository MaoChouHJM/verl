import os

def check_model_api_status(model_api_address, model_api_port):
    """
    执行curl命令检查模型API状态。

    Args:
        model_api_address (str): 模型API的IP地址，可以是逗号分隔的多个IP。
        model_api_port (str): 模型API的端口，可以是逗号分隔的多个端口。
    """
    ips = model_api_address.split(',')
    ports = model_api_port.split(',')

    for ip in ips:
        for port in ports:
            url = f"http://{ip.strip()}:{port.strip()}/v1/models"
            command = f'curl -sf "{url}"'
            print(f"Executing: {command}")
            
            # 执行命令并捕获输出和返回码
            # os.system() 返回命令的退出状态码
            # 如果需要捕获输出，可以使用 subprocess 模块
            result = os.system(command)

            if result == 0:
                print(f"Success: {url} is reachable.")
            else:
                print(f"Failed: {url} is not reachable or returned an error. Exit code: {result}")
            print("-" * 30)

if __name__ == "__main__":
    model_api_address = '10.48.47.84,10.48.47.83'
    model_api_port = '8000,8001,8002,8003'

    check_model_api_status(model_api_address, model_api_port)

