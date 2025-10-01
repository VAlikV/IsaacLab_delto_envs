from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False, enable_cameras=True, enable_livestream=False)
simulation_app = app_launcher.app


import argparse, os, socket, time
from typing import Tuple, Dict, Any

import numpy as np
import torch

from delto_env import DeltoEnv, DeltoEnvCfg

# === если ты используешь мой файл sac2_vision_fusion.py ===
from agents.sac2_vision_fusion import parse_obs   # возвращает (state: (N,S) float32 on device, rgb: (N,H,W,3) uint8 on device?)


# ==========================
# UDP-телеуправление
# ==========================
class UDPActionSource:
    """Читает действия из UDP. Возвращает torch.Tensor формы (num_envs, act_dim)."""
    def __init__(self, ip="127.0.0.1", port=8081, recv_buf_size=4096, timeout_s=0.001):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, recv_buf_size)
        self.sock.bind((ip, port))
        self.sock.settimeout(timeout_s)
        self.sock.setblocking(False)

    def read(self, base_action: torch.Tensor) -> torch.Tensor:
        """base_action: (num_envs, act_dim) — сюда запишем новое действие для env 0, остальные — копия/нулевые."""
        act = base_action.clone()
        try:
            data, address = self.sock.recvfrom(1024)

            data = list(map(float, (str(data)[3:-2].split(", "))))

            # print(data)
            act[:,0:20] = torch.tensor(data)

            # опционально можно «растранслировать» на все env:
            # for i in range(act.shape[0]): act[i, :k] = act[0, :k]
        except (BlockingIOError, socket.timeout, ValueError, UnicodeDecodeError):
            # нет пакетов или мусор — оставляем base_action как есть
            pass
        return act


# ==========================
# Утилита: to_numpy CPU
# ==========================
def _to_numpy_cpu(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu").numpy()
    return np.asarray(x)


# ==========================
# Рекордер траекторий
# ==========================
class TrajRecorder:
    """
    Пишет экспертные траектории в NPZ (uint8 для RGB).
    Для больших записей используйте save_every_steps, чтобы не съедать RAM.
    """
    def __init__(self, save_path: str, save_every_steps: int = 2000):
        self.save_path = save_path
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        self.save_every_steps = save_every_steps
        self.buff: Dict[str, list] = {
            "obs": [],            # state (N,S) float32
            "rgb": [],            # (N,H,W,3) uint8
            "actions": [],        # (N,A) float32
            "rewards": [],        # (N,) float32
            "next_obs": [],
            "next_rgb": [],
            "dones": [],          # (N,) float32 (1 если ep окончен и НЕ бутстрэпим)
            "terminated": [],     # (N,) float32 (истинный терминал)
            "truncated": [],      # (N,) float32 (таймаут/усечение)
            "episode_id": [],     # (N,) int64
            "step": [],           # (N,) int64
            "timestamp": [],      # (N,) float64 (unix time)
        }
        self._flush_count = 0
        self._global_step = 0
        self._episode_counters = {}  # env_id -> episode_id

    def _ensure_ep(self, num_envs: int):
        if not self._episode_counters:
            for i in range(num_envs):
                self._episode_counters[i] = 0

    def _maybe_flush(self):
        if self._global_step % self.save_every_steps == 0 and self._global_step > 0:
            part_path = self.save_path.replace(".npz", f".part{self._flush_count:03d}.npz")
            self._save_npz(part_path)
            self._flush_count += 1
            # очищаем память
            for k in self.buff:
                self.buff[k].clear()

    def _save_npz(self, path: str):
        # конкатенация по time (stack списков)
        npz = {k: np.concatenate([_to_numpy_cpu(x) for x in self.buff[k]], axis=0) if len(self.buff[k]) > 0 else
               (np.zeros((0,), dtype=np.float32) if k not in ["episode_id", "step", "timestamp", "rgb", "next_rgb", "dones", "terminated", "truncated"]
                else np.zeros((0,), dtype=np.uint8))
               for k in self.buff.keys()}
        # типы
        if npz["episode_id"].size > 0:
            npz["episode_id"] = npz["episode_id"].astype(np.int64)
        if npz["step"].size > 0:
            npz["step"] = npz["step"].astype(np.int64)
        if npz["timestamp"].size > 0:
            npz["timestamp"] = npz["timestamp"].astype(np.float64)
        if npz["dones"].size > 0:
            npz["dones"] = npz["dones"].astype(np.float32)
        if npz["terminated"].size > 0:
            npz["terminated"] = npz["terminated"].astype(np.float32)
        if npz["truncated"].size > 0:
            npz["truncated"] = npz["truncated"].astype(np.float32)
        # rgb уже uint8, obs/actions/rewards — float32
        np.savez_compressed(path, **npz)
        print(f"[Saved] {path}")

    def add_step(
        self,
        obs_state: torch.Tensor,          # (N,S) on device
        obs_rgb_uint8: torch.Tensor,      # (N,H,W,3) uint8 on device/CPU
        actions: torch.Tensor,            # (N,A)
        rewards: torch.Tensor,            # (N,) or (N,1)
        next_state: torch.Tensor,
        next_rgb_uint8: torch.Tensor,
        dones: torch.Tensor,              # (N,) bool/float — обычно env.dones
        info: Dict[str, Any],
        step_idx: int
    ):
        N = actions.shape[0]
        self._ensure_ep(N)

        # разделяем terminated и truncated
        terminated = None
        truncated = None
        if isinstance(info, dict):
            terminated = info.get("terminated", None)
            truncated  = info.get("time_outs", info.get("truncated", None))

        def _to1d(x, default=None):
            if isinstance(x, torch.Tensor):
                x = x.detach().to("cpu")
                if x.ndim > 1:
                    x = x.squeeze(-1)
                return x.to(torch.float32).numpy()
            if x is None:
                if default is None:
                    return np.zeros((N,), dtype=np.float32)
                return np.full((N,), default, dtype=np.float32)
            arr = np.asarray(x)
            if arr.ndim > 1:
                arr = arr.squeeze(-1)
            return arr.astype(np.float32)

        dones_f = _to1d(dones)
        term_f  = _to1d(terminated, default=0.0)
        trunc_f = _to1d(truncated,  default=0.0)

        # episode ids
        ep_ids = np.zeros((N,), dtype=np.int64)
        for i in range(N):
            ep_ids[i] = self._episode_counters[i]
            # если эпизод реально закончился, увеличим счётчик
            if term_f[i] > 0.5:
                self._episode_counters[i] += 1

        # складываем (всё в CPU/np)
        ts = np.full((N,), time.time(), dtype=np.float64)
        self.buff["obs"].append(_to_numpy_cpu(obs_state))
        self.buff["rgb"].append(_to_numpy_cpu(obs_rgb_uint8))
        self.buff["actions"].append(_to_numpy_cpu(actions))
        self.buff["rewards"].append(_to_numpy_cpu(rewards).reshape(N, 1))
        self.buff["next_obs"].append(_to_numpy_cpu(next_state))
        self.buff["next_rgb"].append(_to_numpy_cpu(next_rgb_uint8))
        self.buff["dones"].append(dones_f.reshape(N, 1))
        self.buff["terminated"].append(term_f.reshape(N, 1))
        self.buff["truncated"].append(trunc_f.reshape(N, 1))
        self.buff["episode_id"].append(ep_ids.reshape(N, 1))
        self.buff["step"].append(np.full((N, 1), step_idx, dtype=np.int64))
        self.buff["timestamp"].append(ts.reshape(N, 1))

        self._global_step += 1
        self._maybe_flush()

    def finalize(self):
        # сохранить финальный кусок (или весь, если флашей не было)
        base = self.save_path if self._flush_count == 0 else self.save_path.replace(".npz", f".part{self._flush_count:03d}.npz")
        self._save_npz(base)
        print(f"[Done] Trajectories saved to {base}")


# ==========================
# Основная логика записи
# ==========================
def record_manual_trajectories(
    env,
    action_source: UDPActionSource,
    max_steps: int = 300,
    save_path: str = "trajectories/manual_demo.npz",
    save_every_steps: int = 2000
):
    recorder = TrajRecorder(save_path, save_every_steps=save_every_steps)

    # Reset
    obs_dict = env.reset()[0]
    state, rgb = parse_obs(obs_dict)   # state: (N,S), rgb: (N,H,W,3) uint8 (может быть на GPU)
    # подготовим action-тензор
    try:
        act_dim = env.action_space.shape[1]
    except Exception:
        act_dim = env.single_action_space.shape[1]
    actions = torch.zeros((env.num_envs, act_dim), dtype=torch.float32, device=state.device)

    for t in range(max_steps):
        # читаем управление
        actions = action_source.read(actions)

        # Шаг среды
        next_obs_dict, rewards, dones, info, _ = env.step(actions)

        # Парсим следующее наблюдение
        next_state, next_rgb = parse_obs(next_obs_dict)

        # Приводим всё к CPU/np и сохраняем (rgb — uint8 NHWC)
        recorder.add_step(
            obs_state=state.to("cpu"),
            obs_rgb_uint8=(rgb if rgb.dtype == torch.uint8 else (rgb.clamp(0, 255).to(torch.uint8))).to("cpu"),
            actions=actions.to("cpu"),
            rewards=rewards.to("cpu"),
            next_state=next_state.to("cpu"),
            next_rgb_uint8=(next_rgb if next_rgb.dtype == torch.uint8 else (next_rgb.clamp(0, 255).to(torch.uint8))).to("cpu"),
            dones=dones.to("cpu"),
            info=info,
            step_idx=t
        )

        # Переход
        state, rgb = next_state, next_rgb

        # (опционально) Прерывание, если все среды завершились
        if bool(dones.all()):
            # Можно сразу новый эпизод продолжать — мы учли episode_id внутри recorder
            obs_dict = env.reset()[0]
            state, rgb = parse_obs(obs_dict)

        print(t)

    recorder.finalize()


# ==========================
# Entry point
# ==========================
if __name__ == "__main__":

    # Среда
    cfg = DeltoEnvCfg()
    env = DeltoEnv(cfg, render_mode=None)

    # Источник действий по UDP
    action_src = UDPActionSource(ip="192.168.68.122", port=8081, recv_buf_size=4096, timeout_s=0.001)

    # Запись
    os.makedirs("trajectories", exist_ok=True)
    record_manual_trajectories(
        env,
        action_src,
        max_steps=3_000,
        save_path="trajectories/manual_demo.npz",
        save_every_steps=3_000,  # под себя
    )
