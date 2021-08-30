import pylrc
from datetime import datetime
from time import time, sleep
from ffpyplayer.player import MediaPlayer


def parse_lrc(lrc_file):
    LENGTH_STRING = '[length:'
    seconds = None

    with open(lrc_file, 'r') as f:
        for line in f.readlines():
            if line.startswith(LENGTH_STRING):
                line = line.replace(LENGTH_STRING, '')
                timecode = line.strip().replace(']', '')
                t = datetime.strptime(timecode, "%M:%S.%f")
                seconds = t.minute * 60 + t.second + 0.000001 * t.microsecond

    with open(lrc_file, 'r') as f:
        lrc_string = ''.join(f.readlines())

    return pylrc.parse(lrc_string), seconds


class Lyrics:
    def __init__(self, lrc_file):
        self._subs, self._duration = parse_lrc(lrc_file)

    def __len__(self):
        return len(self._subs)

    def get_times_and_texts(self):
        times = [sub.time for sub in self._subs]
        texts = [sub.text for sub in self._subs]
        if self._duration is not None:
            times.append(self._duration)
            texts.append('[EOC]')
        return times, texts

    def __getitem__(self, index):
        if index < len(self):
            text = self._subs[index].text
            time = self._subs[index].time
            if index < len(self) - 1:
                next_time = self._subs[index+1].time
            else:
                if self._duration is not None:
                    next_time = self._duration
                else:
                    next_time = time
        else:
            text, time, next_time = None, None, None

        return text, time, next_time


class ContentPlayer:
    EPSILON = 1.e-6

    def __init__(self, content):
        self._content = content
        self._time = -1.
        self._idx = 0
        self._item, self._time, self._next_time = self._content[0]
        self._start_time = None
        self._is_playing = False
        self._paused_delta = None
        self._paused_time = None

    def start(self):
        self._is_playing = True
        self._start_time = time()

    def current_content(self):
        if self._is_playing:
            cur_time = time() - self._start_time
            is_changed = False
            if cur_time < self._time:
                part = cur_time / (self._time + self.EPSILON)
                return '[WFS]', part, is_changed
            if self._next_time <= cur_time:
                is_changed = True
                self._idx += 1
                self._item, self._time, self._next_time = self._content[self._idx]
            if self._time:
                part = (cur_time - self._time) / (self._next_time - self._time + self.EPSILON)
            else:
                part = None
                self.stop()
            return self._item, part, is_changed
        else:
            return None, None, False

    def stop(self):
        self._is_playing = False
        self._idx = 0
        self._start_time = None

    def pause(self):
        if self._is_playing:
            self._paused_time = time()
            self._is_playing = False
            self._paused_delta = self._paused_time - self._start_time

    def resume(self):
        if not self._is_playing:
            cur_time = time()
            self._start_time = cur_time - self._paused_delta
            self._is_playing = True


def main():
    lrc = Lyrics('believer.lrc')
    player_ = MediaPlayer('believer.mp3')
    player = ContentPlayer(lrc)
    player.start()
    while True:
        item, part, is_changed = player.current_content()
        print(item, part)
        sleep(0.05)
    player.stop()


if __name__ == '__main__':
    main()
