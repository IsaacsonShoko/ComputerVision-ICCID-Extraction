#!/usr/bin/env bash
# stop script on error
set -e

# Check to see if root CA file exists, download if not
if [ ! -f ./root-CA.crt ]; then
  printf "\nDownloading AWS IoT Root CA certificate from AWS...\n"
  curl https://www.amazontrust.com/repository/AmazonRootCA1.pem > root-CA.crt
fi

CWD=`pwd`

# install AWS Device SDK for NodeJS if not already installed + pubsub sample
if [ ! -d ./aws-iot-device-sdk-js-v2 ]; then
  printf "\nInstalling AWS SDK...\n"
  git clone https://github.com/aws/aws-iot-device-sdk-js-v2.git --recursive
  cd aws-iot-device-sdk-js-v2
  npm install
  # samples require their own install
  cd samples/node/mqtt/mqtt5_x509
  npm install
  cd $CWD
fi

# run pub/sub sample app using certificates downloaded in package
printf "\nRunning pub/sub sample application...\n"
node aws-iot-device-sdk-js-v2/samples/node/mqtt/mqtt5_x509/dist/index.js --endpoint a3pxon32gjm429-ats.iot.eu-north-1.amazonaws.com --key raspberrypi-alpha.private.key --cert raspberrypi-alpha.cert.pem --client_id sdk-nodejs-v2 --topic sdk/test/js