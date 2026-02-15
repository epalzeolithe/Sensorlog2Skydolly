import numpy as np
import pandas as pd
from vispy import app, scene


import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from vispy import app, scene
from vispy.visuals.transforms import MatrixTransform
from vispy.geometry import create_box, create_cone


af = pd.read_csv("sensorlog.csv",low_memory=False)
print ("Taille sensorlog.csv :",af.shape)
df = af.iloc[::5].reset_index(drop=True)
print ("Taille après cut :",df.shape)

df = df.rename(columns={'loggingSample(N)': 'Milliseconds', 'loggingTime(txt)': 'UTC','locationLatitude(WGS84)': 'Latitude','locationLongitude(WGS84)': 'Longitude','locationAltitude(m)': 'Altitude','locationSpeed(m/s)': 'Speed', 'motionYaw(rad)': 'YawRad','motionPitch(rad)': 'PitchRad', 'motionRoll(rad)': 'RollRad', 'locationTrueHeading(°)': 'Heading'})

df["Altitude"] = df["Altitude"].rolling(window=100, center=True).mean()
df["Latitude"] = df["Latitude"].rolling(window=100, center=True).mean()
df["Longitude"] = df["Longitude"].rolling(window=100, center=True).mean()
df["Speed"] = df["Speed"].rolling(window=100, center=True).mean()

df['G'] = round(np.sqrt(
    df['accelerometerAccelerationX(G)']**2 +
    df['accelerometerAccelerationY(G)']**2 +
    df['accelerometerAccelerationZ(G)']**2
),1)

df["Milliseconds"] = df["Milliseconds"].rolling(window=100, center=True).mean()
df["s"]=df['Milliseconds']/1000
df['fpm'] = np.gradient(df['Altitude'], df['s'])


# -------- Conversion angles d'Euler -> vecteurs 3 axes --------
# ==========================
# 2️⃣  Rotations iOS
# ==========================

def Rz(yaw):
    return np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])

def Rx(pitch):
    return np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch),  np.cos(pitch)]
    ])

def Ry(roll):
    return np.array([
        [ np.cos(roll), 0, np.sin(roll)],
        [0, 1, 0],
        [-np.sin(roll), 0, np.cos(roll)]
    ])

#calibration initiale
enligne_30G=24480 # mise en ligne 30G
R0 = Rz(df['YawRad'][enligne_30G]) @ Rx(df['PitchRad'][enligne_30G]) @ Ry(df['RollRad'][enligne_30G])
R0 = np.linalg.inv(R0)

def forward_vector(pitch, roll, yaw):
    R = Rz(yaw) @ Rx(pitch) @ Ry(roll)
    R = R0 @ R
    z_axis = R[:, 2]
    return z_axis  # caméra regarde vers Z

def up_vector(pitch, roll, yaw):
    R = Rz(yaw) @ Rx(pitch) @ Ry(roll)
    R = R0 @ R
    y_axis = R[:, 1]
    return y_axis  # caméra regarde vers -Z

def right_vector(pitch, roll, yaw):
    R = Rz(yaw) @ Rx(pitch) @ Ry(roll)
    R = R0 @ R
    x_axis = R[:, 0]
    return x_axis  # caméra regarde vers -Z

# -------- Création de la scène VisPy --------
canvas = scene.SceneCanvas(keys='interactive', show=True, title="Axes 3D Euler")
view = canvas.central_widget.add_view()
view.camera = 'turntable'
axis = scene.visuals.XYZAxis(parent=view.scene)
#une ligne rouge → axe X
#une ligne verte → axe Y
#une ligne bleue → axe Z

#cube = scene.visuals.Box(width=0.1, height=0.1, depth=0.1, color='skyblue', edge_color='black', parent=view.scene)
#transform = MatrixTransform()
#cube.transform = transform

# Vecteurs initiaux
line_forward = scene.visuals.Line(pos=np.array([[0, 0, 0], [0, 0, 1]]),color='white', width=8, parent=view.scene)
line_tail = scene.visuals.Line(pos=np.array([[0, 0, 0], [0, 0, -1]]),color='white', width=8, parent=view.scene)
line_up = scene.visuals.Line(pos=np.array([[0, 0, 0], [0, 1, 0]]),color='green',width=4,parent=view.scene)
line_aile_gauche = scene.visuals.Line(pos=np.array([[0, 0, 0], [1, 0, 0]]),color='yellow',width=4,parent=view.scene)
line_aile_droite = scene.visuals.Line(pos=np.array([[0, 0, 0], [-1, 0, 0]]),color='yellow',width=4,parent=view.scene)
line_cockpit = scene.visuals.Line(pos=np.array([[0, 0, 0], [0, 1, 0]]),color='green',width=4,parent=view.scene)
# Texte pour numéro de frame
frame_text = scene.visuals.Text(text="Frame: 0", color='white', font_size=10,pos=(10, 10), parent=canvas.scene, anchor_x='left', anchor_y='bottom')
utc_text = scene.visuals.Text(text="UTC: 0", color='white', font_size=10,pos=(200, 10), parent=canvas.scene, anchor_x='left', anchor_y='bottom')
speed_text = scene.visuals.Text(text="Speed(km/h): 0", color='white', font_size=10,pos=(10,28), parent=canvas.scene, anchor_x='left', anchor_y='bottom')
altitude_text = scene.visuals.Text(text="Altitude(feet): 0", color='white', font_size=10,pos=(10, 46), parent=canvas.scene, anchor_x='left', anchor_y='bottom')
fpm_text = scene.visuals.Text(text="Altitude(feet): 0", color='white', font_size=10,pos=(10, 64), parent=canvas.scene, anchor_x='left', anchor_y='bottom')
heading_text = scene.visuals.Text(text="Heading: 0", color='white', font_size=10,pos=(10, 82), parent=canvas.scene, anchor_x='left', anchor_y='bottom')
assiette_text = scene.visuals.Text(text="Assiette: 0", color='white', font_size=10,pos=(10, 100), parent=canvas.scene, anchor_x='left', anchor_y='bottom')
inclinaison_text = scene.visuals.Text(text="Inclinaison: 0", color='white', font_size=10,pos=(10, 118), parent=canvas.scene, anchor_x='left', anchor_y='bottom')
g_text = scene.visuals.Text(text="G: 0", color='white', font_size=10,pos=(10, 136), parent=canvas.scene, anchor_x='left', anchor_y='bottom')


# -------- Fonction de mise à jour --------
index = int(28    *60*100/5 ) # minutes/secondes à 100hz/5
index = 24480
def update(event):
    frame=event.count+index

    if frame>len(df)-2:
        timer.stop();

    pitch = df['PitchRad'][frame]
    roll = df['RollRad'][frame]
    yaw = df['YawRad'][frame]
    alt= int(df['Altitude'][frame]* 3.28084)
    speed = int(df['Speed'][frame]*3.6)
    heading = round(df['Heading'][frame],1)
    #yaw=yaw-heading/360*2*math.pi
    date=df['UTC'][frame]

    fwd = forward_vector(pitch, roll, yaw)
    line_forward.set_data(pos=np.array([[0, 0, 0], fwd/2]))
    line_tail.set_data(pos=np.array([[0, 0, 0], -fwd]))

    up = up_vector(pitch, roll, yaw)/2
    line_up.set_data(pos=np.array([[0, 0, 0], up]))

    aile_gauche = right_vector(pitch, roll, yaw)
    line_aile_gauche.set_data(pos=np.array([[0, 0, 0], aile_gauche]))
    line_aile_droite.set_data(pos=np.array([[0, 0, 0], -aile_gauche]))

    line_cockpit.set_data(pos=np.array([fwd/2,up]))

    #calcul des angles pitch=assiette, bank=inclinaison, heading pour le 3ème angle
    #assiette, pitch = angle entre fwd blanc et plan horizontal
    v = np.array([0, 1, 0]) # y vert
    assiette = round(90-(np.arccos(np.dot(fwd, v) / (np.linalg.norm(fwd) * np.linalg.norm(v))))/math.pi*180,1)
    assiette_text.text = f"Assiette : {assiette}"

    v = np.array([0, 1, 0]) # y vert
    inclinaison = round(90-(np.arccos(np.dot(aile_gauche, v) / (np.linalg.norm(aile_gauche) * np.linalg.norm(v))))/math.pi*180,1)
    inclinaison_text.text = f"Inclinaison : {inclinaison}"

    #TO DO, fix inclinaison et assiette en vol dos

    frame_text.text = f"Frame : {frame}"
    speed_text.text = f"Speed(km/h): {speed}"
    altitude_text.text = f"Altitude(feet) : {alt}"
    heading_text.text = f"Heading : {heading}"
    utc_text.text = f"Time : {date}"
    g_text.text = f"G : {df['G'][frame]}"
    fpm_text.text = f"Vario : {round(df['fpm'][frame]*60*3.28084/5)}"
# Timer pour animation
timer = app.Timer(interval=0.05, connect=update, start=True)  # 20 FPS

# -------- Lancement de l'application --------
if __name__ == '__main__':
    app.run()
