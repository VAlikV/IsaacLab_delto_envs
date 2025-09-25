import numpy as np

def rpy_to_quat(roll, pitch, yaw):
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qw, qx, qy, qz])  # w, x, y, z

def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion (w, x, y, z).
    Assumes R is a proper rotation (orthonormal, det ~ 1).
    """
    R = np.asarray(R, dtype=float)
    assert R.shape == (3, 3)
    # (опционально) слегка ортонормируем на всякий случай
    # U, _, Vt = np.linalg.svd(R); R = U @ Vt

    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    else:
        # Выбираем наибольший диагональный элемент
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            w = (R[2,1] - R[1,2]) / s
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s

    q = np.array([w, x, y, z], dtype=float)
    # нормализуем (на случай накопленных ошибок)
    q /= np.linalg.norm(q)
    return q

def rot_x(theta):
    """Поворот вокруг оси X на угол theta (рад)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])

def rot_y(theta):
    """Поворот вокруг оси Y на угол theta (рад)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])

def rot_z(theta):
    """Поворот вокруг оси Z на угол theta (рад)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

def quaternion_multiply(q1, q2):
        """
        Умножение кватернионов q1 * q2
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return np.array([w, x, y, z])

x = 90*np.pi/180
y = 0*np.pi/180
z = 12.5*np.pi/180

qx = np.array([np.cos(x/2), np.sin(x/2), 0, 0])
qy = np.array([np.cos(y/2), 0, np.sin(y/2), 0])
qz = np.array([np.cos(z/2), 0, 0, np.sin(z/2)])

q = quaternion_multiply(quaternion_multiply(qz, qy), qx)

print(rpy_to_quat(0*np.pi/180, -75*np.pi/180, -90*np.pi/180))
print(q)
# print(rpy_to_quat(30*np.pi/180, -75*np.pi/180, -90*np.pi/180))
