

import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from vispy import app, scene
from vispy.visuals.transforms import MatrixTransform
from vispy.geometry import create_box, create_cone
from vispy.visuals.filters import ShadingFilter

from stl import mesh
from vispy.visuals.transforms import STTransform


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
#enligne_30G=24480 # mise en ligne 30G

#calibration auto # recherche quand première fois à 90km/h
vitesse_min = 80 # vitesse mise en ligne
mask = df['Speed'] > vitesse_min/3.6
enligne_devol= mask.idxmax()
R0 = Rz(df['YawRad'][enligne_devol]) @ Rx(df['PitchRad'][enligne_devol]) @ Ry(df['RollRad'][enligne_devol])
R0 = np.linalg.inv(R0)
print("En ligne de vol à", vitesse_min, "km/h Frame#",enligne_devol," @ ",df['UTC'][enligne_devol])

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

def rotation_matrix(pitch, roll, yaw):
    return Rz(yaw) @ Rx(pitch) @ Ry(roll)
# -------- Création de la scène VisPy --------
canvas = scene.SceneCanvas(keys='interactive', show=True, title="Plane Attitude Player")
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(fov=45, distance=2.5, up='+y')
#view.camera = 'arcball'
axis = scene.visuals.XYZAxis(parent=view.scene)

# ==========================
# 4️⃣ Construction avion
# ==========================

plane = scene.Node(parent=view.scene)
transform = MatrixTransform()
plane.transform = transform

stl_mesh = mesh.Mesh.from_file("F4U.STL")
vertices = stl_mesh.vectors.reshape(-1,3).astype('float32')
faces = np.arange(len(vertices)).reshape(-1,3).astype('uint32')
scale = np.max(np.abs(vertices))
if scale > 0:
    vertices /= scale
# -------- OBJET STL --------
f4u = scene.visuals.Mesh(vertices=vertices,faces=faces,color='lightgray',parent=plane)
f4u.transform = MatrixTransform()
f4u.transform.translate((-0.5, -0.05,-0.3))
f4u.transform.rotate(180,(0, 1, 0))
f4u.transform.scale((2,2,2))
shading = ShadingFilter(shading='smooth',light_dir=(0,-1,0),ambient_light=(0.3,0.3,0.3),diffuse_light=(0.4,0.4,0.4),specular_light=(1,1,1),shininess=50)
f4u.attach(shading)

# Vecteurs initiaux
line_forward = scene.visuals.Line(pos=np.array([[0, 0, 0], [0, 0, 1]]),color='white', width=1, parent=view.scene)
line_tail = scene.visuals.Line(pos=np.array([[0, 0, 0], [0, 0, -1]]),color='white', width=1, parent=view.scene)
line_up = scene.visuals.Line(pos=np.array([[0, 0, 0], [0, 1, 0]]),color='yellow',width=1,parent=view.scene)
line_aile = scene.visuals.Line(pos=np.array([[0, 0, 0], [1, 0, 0]]),color='white',width=1,parent=view.scene)
line_cockpit = scene.visuals.Line(pos=np.array([[0, 0, 0], [0, 1, 0]]),color='yellow',width=1,parent=view.scene)
#line_forward_vertical = scene.visuals.Line(pos=np.array([[0, 0, 0], [0, 1, 0]]),color='purple',width=1,parent=view.scene)

# Texte pour numéro de frame
frame_text = scene.visuals.Text(text="Frame: 0", color='white', font_size=10,pos=(10, 10), parent=canvas.scene, anchor_x='left', anchor_y='bottom')
utc_text = scene.visuals.Text(text="UTC: 0", color='white', font_size=10,pos=(200, 10), parent=canvas.scene, anchor_x='left', anchor_y='bottom')
speed_text = scene.visuals.Text(text="Speed(km/h): 0", color='white', font_size=10,pos=(10,28), parent=canvas.scene, anchor_x='left', anchor_y='bottom')
altitude_text = scene.visuals.Text(text="Altitude(feet): 0", color='white', font_size=10,pos=(10, 46), parent=canvas.scene, anchor_x='left', anchor_y='bottom')
#fpm_text = scene.visuals.Text(text="Altitude(feet): 0", color='white', font_size=10,pos=(10, 64), parent=canvas.scene, anchor_x='left', anchor_y='bottom')
heading_text = scene.visuals.Text(text="Heading: 0", color='white', font_size=10,pos=(10, 82), parent=canvas.scene, anchor_x='left', anchor_y='bottom')
assiette_text = scene.visuals.Text(text="Assiette: 0", color='white', font_size=10,pos=(10, 100), parent=canvas.scene, anchor_x='left', anchor_y='bottom')
inclinaison_text = scene.visuals.Text(text="Inclinaison: 0", color='white', font_size=10,pos=(10, 118), parent=canvas.scene, anchor_x='left', anchor_y='bottom')
g_text = scene.visuals.Text(text="G: 0", color='white', font_size=20,pos=(10, 136), parent=canvas.scene, anchor_x='left', anchor_y='bottom')


# -------- Fonction de mise à jour --------
index = int(28    *60*100/5 ) # minutes/secondes à 100hz/5
#index = 24480 #takeoff
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
    line_forward.set_data(pos=np.array([[0, 0, 0], fwd]))
    line_tail.set_data(pos=np.array([[0, 0, 0], -fwd]))
    up = up_vector(pitch, roll, yaw)
    line_up.set_data(pos=np.array([[0, 0, 0], up]))
    aile_gauche = right_vector(pitch, roll, yaw)
    line_aile.set_data(pos=np.array([-aile_gauche*1.2, aile_gauche*1.2]))
    line_cockpit.set_data(pos=np.array([fwd/2,up]))

    # angle vecteur forward avec projection au sol
    assiette= round(((np.arctan(fwd[1]/math.sqrt(fwd[0]**2 +fwd[2]**2)))/math.pi*180),1)
    assiette_text.text = f"Assiette : {assiette}"

    #calcul vecteur orthogonal à l'axe longitudinal, et dans le plan vertical
    x=fwd[0];y=fwd[1];z=fwd[2]
    v = np.array([-x * y / math.sqrt(x ** 2 + z ** 2), math.sqrt(x ** 2 + z ** 2), -z * y / math.sqrt(x ** 2 + z ** 2)]) # vecteur orthogonal à fwd, dans le plan vertical
    #line_forward_vertical.set_data(pos=np.array([[0, 0, 0], v])) # affiche ce vecteur

    # calcul inclinaison, produit scalaire et vectoriel
    axis = np.array([0, 0, 1])  # axe autour duquel mesurer le signe
    # angle
    dot = np.dot(up, v)
    cross = np.cross(up, v)
    angle = np.arctan2(np.dot(cross, fwd), dot)
    angle_deg = round(np.degrees(angle),1)
    inclinaison_text.text = f"Inclinaison : {angle_deg}"
    #ancienne methote mais angle non signé
    # inclinaison = round((np.arccos(np.dot(up, v) / (np.linalg.norm(up) * np.linalg.norm(v))))/math.pi*180,1) #calcul angle entre les 2 vecteurs
    # inclinaison_text.text = f"Inclinaison : {inclinaison}"

    #rotation du model STL
    R = rotation_matrix(pitch, roll, yaw)
    R = R0 @ R
    M = np.eye(4)
    M[:3, :3] = R.T
    transform.matrix = M

    g_text.text = f"G : {df['G'][frame]}"

    frame_text.text = f"Frame : {frame}"
    speed_text.text = f"Speed(km/h): {speed}"
    altitude_text.text = f"Altitude(feet) : {alt}"
    heading_text.text = f"Heading : {heading}"
    utc_text.text = f"Time : {date}"
    #fpm_text.text = f"Vario : {round(df['fpm'][frame]*60*3.28084/5)}"
# Timer pour animation
timer = app.Timer(interval=0.05, connect=update, start=True)  # 20 FPS

# -------- Lancement de l'application --------
if __name__ == '__main__':
    app.run()
