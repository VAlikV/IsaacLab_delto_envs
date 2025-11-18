# IsaacSim_delto_envs
IsaacLab envs with tesolo delto for RL and control experiments

## Интеграция

1. Переместить файлы из `robots/` в `<директория IsaacLab>/robots/dg5f_right/`

2. Переместить файлы из `envs/` в `<директория IsaacLab>/scripts/my_examples/` (что-то типо)

3. Переместить файлы из `tasks/` в `<директория IsaacLab>/source/isaaclab_tasks/isaaclab_tasks/direct`

## Запуск среды

```bash
./isaaclab.sh -p scripts/my_examples/tesolo_delto_UR_env/setup_env.py 
```

## Запуск задачи

```bash
# Train
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task Isaac-Tesolo-Delto-Direct-D --headless --num_envs 2048 --wandb-entity rubitek_dextrous --wandb-project-name RubetekDextorousDirectEnv --wandb-name UR10-Tessolo-v21-ForceRew --track

# Val
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task Isaac-Tesolo-Delto-Direct-D --num_envs 3
```
## Загрузка весов

Распаковать `reorient.zip` в `<директория IsaacLab>/logs/rl_games/`

## Описание

1. `envs/` - среды isaaclab

2. `robots/` - usd файлы роботов

3. `scripts` - скрипты для управленяи
