# Record plane attitude at 50hz usine simple iPhone
# Use of Sensorlog app, available on App Store
# Convert the sensorlog.csv to SkyDolly Flight Recorder Format
# Replay using MSFS

# Version 0.1

import pandas as pd
import math
import numpy as np
from datetime import datetime

df = pd.read_csv("sensorlog.csv",low_memory=False)
print ("Taille sensorlog.csv :",df.shape)
df.drop(['locationSpeedAccuracy(m/s)', 'locationCourseAccuracy(°)', 'locationTimestamp_since1970(s)','locationVerticalAccuracy(m)', 'locationHorizontalAccuracy(m)', 'locationFloor(Z)', 'locationHeadingTimestamp_since1970(s)', 'locationHeadingX(µT)', 'locationHeadingY(µT)', 'locationHeadingZ(µT)', 'locationHeadingAccuracy(°)', 'accelerometerTimestamp_sinceReboot(s)', 'accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)', 'accelerometerAccelerationZ(G)', 'gyroTimestamp_sinceReboot(s)', 'gyroRotationX(rad/s)', 'gyroRotationY(rad/s)', 'gyroRotationZ(rad/s)', 'magnetometerTimestamp_sinceReboot(s)', 'magnetometerX(µT)', 'magnetometerY(µT)', 'magnetometerZ(µT)', 'motionTimestamp_sinceReboot(s)', 'motionRotationRateX(rad/s)', 'motionRotationRateY(rad/s)', 'motionRotationRateZ(rad/s)', 'motionUserAccelerationX(G)', 'motionUserAccelerationY(G)', 'motionUserAccelerationZ(G)', 'motionAttitudeReferenceFrame(txt)', 'motionQuaternionX(R)', 'motionQuaternionY(R)', 'motionQuaternionZ(R)', 'motionQuaternionW(R)', 'motionGravityX(G)', 'motionGravityY(G)', 'motionGravityZ(G)', 'motionMagneticFieldX(µT)', 'motionMagneticFieldY(µT)', 'motionMagneticFieldZ(µT)', 'motionMagneticFieldCalibrationAccuracy(Z)', 'activityTimestamp_sinceReboot(s)', 'activity(txt)', 'activityActivityConfidence(Z)', 'activityActivityStartDate(txt)', 'altimeterTimestamp_sinceReboot(s)', 'altimeterReset(bool)', 'motionHeading(°)', 'deviceID(txt)', 'label(N)'], axis=1, inplace=True)
df = df.rename(columns={'loggingSample(N)': 'Milliseconds', 'loggingTime(txt)': 'UTC','locationLatitude(WGS84)': 'Latitude','locationLongitude(WGS84)': 'Longitude','locationAltitude(m)': 'Altitude','locationSpeed(m/s)': 'Speed', 'motionPitch(rad)': 'PitchRad', 'motionRoll(rad)': 'BankRad', 'locationTrueHeading(°)': 'Heading'})

df['UTC'] = pd.to_datetime(df['UTC'])
print("Fréquence sensorlog.csv : ",round(1/(df["UTC"][1001]-df["UTC"][0]).total_seconds()*1000), "Hz")

pitch_calibration = df['PitchRad'][1]/math.pi*180
bank_calibration = df['BankRad'][1]/math.pi*180
altitude_calibration = 30

df['Pitch']=(df['PitchRad']/math.pi*180-pitch_calibration)
df['Bank']=(df['BankRad']/math.pi*180-bank_calibration)
df['Speed']=df['Speed']*3.6
df["Speed"] = df["Speed"].astype(int)
df["Pitch"] = df["Pitch"].astype(int)
df["Bank"] = df["Bank"].astype(int)
df["Altitude"] = df["Altitude"].astype(int)*3.28084-altitude_calibration # en feet
df['Latitude']=df['Latitude'].round(6)
df['Longitude']=df['Longitude'].round(6)
df["MagneticHeading"] = df["locationMagneticHeading(°)"].astype(int)


df["Altitude"] = df["Altitude"].rolling(window=100, center=True).mean()
df["Latitude"] = df["Latitude"].rolling(window=100, center=True).mean()
df["Longitude"] = df["Longitude"].rolling(window=100, center=True).mean()
df["Pitch"] = df["Pitch"].rolling(window=50, center=True).mean()
df["Bank"] = df["Bank"].rolling(window=50, center=True).mean()
df["Heading"] = df["Heading"].rolling(window=50, center=True).mean()
df["MagneticHeading"] = df["MagneticHeading"].rolling(window=50, center=True).mean()


df.drop(['altimeterRelativeAltitude(m)','altimeterPressure(kPa)','locationMagneticHeading(°)'], axis=1, inplace=True)
# -----------------------------
# 4️⃣ Décimation / sous-échantillonnage
factor = 5  # Facteur de réduction
df_dec = df.iloc[::factor].reset_index(drop=True)
print("Format après décimation : ", df_dec.shape)
print("Fréquence Sensorlog après décimation : ",round(1/(df_dec["UTC"][1001]-df_dec["UTC"][0]).total_seconds()*1000), "Hz")

df_dec['Milliseconds']=(df_dec['UTC']-df_dec['UTC'][0])
df_dec['Milliseconds']=(df_dec['Milliseconds'].astype("int64")/1000).astype(int)

fn_labels=['Milliseconds', 'Latitude', 'Longitude', 'Altitude', 'Pitch', 'Bank',
       'GyroHeading', 'TrueHeading', 'MagneticHeading', 'VelocityBodyX',
       'VelocityBodyY', 'VelocityBodyZ', 'AileronPosition', 'ElevatorPosition',
       'RudderPosition', 'ElevatorTrimPosition', 'AileronTrimPercent',
       'RudderTrimPercent', 'FlapsHandleIndex', 'TrailingEdgeFlapsLeftPercent',
       'TrailingEdgeFlapsRightPercent', 'LeadingEdgeFlapsLeftPercent',
       'LeadingEdgeFlapsRightPercent', 'ThrottleLeverPosition1',
       'ThrottleLeverPosition2', 'ThrottleLeverPosition3',
       'ThrottleLeverPosition4', 'PropellerLeverPosition1',
       'PropellerLeverPosition2', 'PropellerLeverPosition3',
       'PropellerLeverPosition4', 'SpoilerHandlePosition',
       'GearHandlePosition', 'WaterRudderHandlePosition', 'BrakeLeftPosition',
       'BrakeRightPosition', 'BrakeParkingPosition', 'LightTaxi',
       'LightLanding', 'LightStrobe', 'LightBeacon', 'LightNav', 'LightWing',
       'LightLogo', 'LightRecognition', 'LightCabin', 'SimulationRate',
       'AbsoluteTime', 'AltitudeAboveGround', 'IsOnGround', 'WindVelocity',
       'WindDirection', 'GForce', 'TouchdownNormalVelocity',
       'WingFlexPercent1', 'WingFlexPercent2', 'WingFlexPercent3',
       'WingFlexPercent4', 'TrueAirspeed', 'IndicatedAirspeed', 'MachAirspeed',
       'GpsGroundSpeed', 'GroundSpeed', 'HeadingIndicator', 'AIPitch',
       'AIBank', 'EngineManifoldPressure1', 'EngineManifoldPressure2',
       'EngineManifoldPressure3', 'EngineManifoldPressure4',
       'TurnCoordinatorBall', 'HsiCDI', 'StallWarning',
       'RotationVelocityBodyX', 'RotationVelocityBodyY',
       'RotationVelocityBodyZ', 'AccelerationBodyX', 'AccelerationBodyY',
       'AccelerationBodyZ']

fn = pd.DataFrame(0, index=range(df_dec.shape[0]), columns=fn_labels)

fn["Pitch"] = -df_dec["Bank"]/3.14
fn["Bank"] = df_dec["Pitch"]
fn["TrueHeading"] = df_dec["Heading"]
fn["Altitude"] = df_dec["Altitude"]
fn['Latitude']=df_dec['Latitude'].round(6)
fn['Longitude']=df_dec['Longitude'].round(6)
fn['Milliseconds']=df_dec['Milliseconds'].astype(int)

#fn['VelocityBodyX']=0
#fn['VelocityBodyY']=0
fn['VelocityBodyZ']=df_dec["Speed"]

fn = fn.iloc[10:] #nettoyage premieres lignes generees sans coordonnees
fn = fn.iloc[:-10] #nettoyage dernieres lignes generees sans coordonnees
fn.to_csv("sensorlog4skydolly.csv", index=False,encoding="utf-8")


