# AWS IoT Core Integration Guide

This document provides steps and sample code to connect the Raspberry Pi to AWS IoT Core and route detection events to Lambda / n8n.

## Overview
- Publish detection messages to MQTT topic `pi-imaging/detections` with QoS=1
- Use X.509 certs for authentication
- Use an IoT Rule to trigger Lambda which forwards to n8n

## Certificates
1. Create IoT thing in AWS Console.
2. Create certificates and download:
   - `device.cert.pem`
   - `device.private.key`
   - `AmazonRootCA1.pem`
3. Put them in `certs/` and set permissions:
```bash
chmod 400 certs/device.private.key
chmod 444 certs/device.cert.pem certs/AmazonRootCA1.pem
```

## Python MQTT publishing example (boto3 / paho alternative)
```python
import ssl
import json
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

client = AWSIoTMQTTClient("raspberrypi-alpha")
client.configureEndpoint("<YOUR_IOT_ENDPOINT>", 8883)
client.configureCredentials("certs/AmazonRootCA1.pem", "certs/device.private.key", "certs/device.cert.pem")
client.configureOfflinePublishQueueing(-1)  # Infinite offline queueing
client.configureDrainingFrequency(2)  # Draining: 2 Hz
client.configureConnectDisconnectTimeout(10)
client.configureMQTTOperationTimeout(5)

client.connect()

payload = {
  "run_id": "20251129_abc123",
  "detections": []
}

client.publish("pi-imaging/detections", json.dumps(payload), 1)
```

## IoT Rule to Lambda
- SQL: `SELECT * FROM 'pi-imaging/detections'`
- Action: Invoke Lambda function `pi_imaging_to_n8n`

Lambda should validate payload and POST to n8n webhook.

## Testing
- Use `mosquitto_pub` with TLS to test publishing.
- Monitor IoT Core metrics and CloudWatch logs for Lambda invocations.

## Security best practices
- Limit policy to required topics and resources rather than `"Resource": "*"`.
- Rotate IAM keys and maintain strict permissions for the Lambda role.

