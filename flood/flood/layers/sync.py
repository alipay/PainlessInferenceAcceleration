# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import time

import torch


class DeviceSyncLayer(torch.nn.Module):
    def __init__(self, dst, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dst = dst
        self.logging = False

    def __call__(self, hidden_states, *args, **kwargs):
        device = torch.device(f'cuda:{self.dst}')
        hidden_states = hidden_states.to(device)
        kwargs['meta'].to(device)

        return hidden_states


class NodeSyncLayer():
    def __init__(self, dst, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dst = dst
        self.logging = False

    def __call__(self, hidden_states, *args, **kwargs):
        # dist.send(hidden_states, 1, group=None)
        return hidden_states


class TaskSyncLayer():
    def __init__(self, idx, pre_task, task, sync_wait_time, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.idx = idx
        self.pre_task = pre_task
        self.task = task
        self.sync_wait_time = sync_wait_time

    def __call__(self, *args, **kwargs):

        if self.idx != 0:
            with self.pre_task.get_lock():
                self.pre_task.value -= 1

        if self.idx == -1:
            return

        tick = 0
        n_task = self.task.value
        tick_time = 0.001
        while n_task >= 1:  # TODO
            time.sleep(tick_time)
            tick += 1
            if n_task >= 2 and tick >= self.sync_wait_time[1] / tick_time:
                break
            elif n_task == 1 and tick >= self.sync_wait_time[0] / tick_time:
                break
            n_task = self.task.value

        with self.task.get_lock():
            self.task.value += 1
