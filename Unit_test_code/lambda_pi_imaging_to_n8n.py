"""
AWS Lambda Function: pi-imaging-to-n8n
======================================

PURPOSE:
--------
This Lambda function acts as a bridge between AWS IoT Core and the n8n webhook.
When the Raspberry Pi publishes a message to the 'pi-imaging/detections' MQTT topic,
AWS IoT Core triggers this Lambda function which then forwards the message to n8n
for workflow processing.

FLOW:
-----
1. Raspberry Pi captures image → uploads to S3
2. Pi publishes message to IoT Core topic: 'pi-imaging/detections'
3. IoT Rule 'pi_imaging_to_n8n' triggers this Lambda
4. Lambda forwards the message to n8n webhook via HTTP POST
5. n8n processes the message (sends to Google Sheets, notifications, etc.)

AWS RESOURCES:
--------------
- Lambda Function Name: pi-imaging-to-n8n
- Lambda ARN: arn:aws:lambda:eu-north-1:253819957157:function:pi-imaging-to-n8n
- IoT Rule Name: pi_imaging_to_n8n
- IoT Rule ARN: arn:aws:iot:eu-north-1:253819957157:rule/pi_imaging_to_n8n
- IoT Topic: pi-imaging/detections

PERMISSIONS REQUIRED:
---------------------
The IoT Rule needs permission to invoke this Lambda. This should be auto-created
when you add the Lambda action in the IoT Rule console. If not, run:

    aws lambda add-permission \
      --function-name pi-imaging-to-n8n \
      --statement-id iot-rule-invoke \
      --action lambda:InvokeFunction \
      --principal iot.amazonaws.com \
      --source-arn arn:aws:iot:eu-north-1:253819957157:rule/pi_imaging_to_n8n

To verify the permission exists:

    aws lambda get-policy --function-name pi-imaging-to-n8n

EXPECTED MESSAGE FORMAT:
------------------------
{
    "s3_key": "MTN/run_20241124_120000/image_001.jpg",
    "s3_url": "https://bucket.s3.amazonaws.com/...",
    "s3_bucket": "your-bucket-name",
    "s3_region": "eu-north-1",
    "run_id": "run_20241124_120000",
    "image_name": "image_001.jpg",
    "image_number": 1,
    "timestamp": "2024-11-24T12:00:00.000Z",
    "service_provider": "MTN",
    "device_id": "raspberrypi-alpha"
}

TESTING:
--------
1. In AWS Console, go to IoT Core → Test → MQTT test client
2. Publish to topic: pi-imaging/detections
3. Message payload: {"test": "hello", "timestamp": "2024-11-24T12:00:00Z"}
4. Check Lambda CloudWatch logs for execution results
5. Check n8n webhook for received data

TROUBLESHOOTING:
----------------
- If Lambda not invoked: Check IoT Rule is enabled and Lambda action is configured
- If n8n not receiving: Check n8n webhook URL is correct and server is accessible
- Check CloudWatch Logs: /aws/lambda/pi-imaging-to-n8n
"""

import json
import urllib.request
import urllib.error

def lambda_handler(event, context):
    """Forward IoT Core messages to n8n webhook"""

    # n8n webhook URL - update this if your webhook changes
    url = "http://84.x.x.x:5678/webhook/72f60ef6-0d33-435c-9452-ec47caf7cb8b"

    # Log incoming event for debugging
    print(f"Received event: {json.dumps(event)}")

    # Event from IoT Rule is already the message payload
    payload = json.dumps(event).encode('utf-8')

    req = urllib.request.Request(
        url,
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            response_body = response.read().decode('utf-8')
            print(f"Success: {response.status} - {response_body[:200]}")
            return {
                'statusCode': response.status,
                'body': response_body
            }
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.reason}")
        return {
            'statusCode': e.code,
            'error': e.reason
        }
    except urllib.error.URLError as e:
        print(f"URL Error: {e.reason}")
        return {
            'statusCode': 500,
            'error': str(e.reason)
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'error': str(e)
        }
