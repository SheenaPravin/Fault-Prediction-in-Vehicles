/* Live telemetry feed: streams the physics-simulated mission record-by-record
 * in real time (dt-scaled), looping continuously so parameter values keep
 * changing live. Speed control multiplies the feed rate. */

"use strict";

const LiveFeed = {
  stream: null,
  cfg: null,
  idx: 0,
  cycle: 0,
  playing: false,
  speed: 1,
  _timer: null,
  _cb: null,
  _cycleCb: null,

  async init(cfg) {
    this.cfg = cfg;
    this.stream = await fetch("live_stream.json").then(r => r.json());
    return this;
  },

  get records() { return this.stream.records; },
  get durationS() { return (this.stream.records.length - 1) * this.cfg.dt; },

  onRecord(cb) { this._cb = cb; },
  onCycle(cb) { this._cycleCb = cb; },

  play() {
    if (this.playing || !this.stream) return;
    this.playing = true;
    if (this._cb) this._cb(this.records[this.idx], this.idx, this.cycle);
    const loop = () => {
      if (!this.playing) return;
      this.idx += 1;
      if (this.idx >= this.records.length) {
        this.idx = 0;
        this.cycle += 1;
        if (this._cycleCb) this._cycleCb(this.cycle);
      }
      if (this._cb) this._cb(this.records[this.idx], this.idx, this.cycle);
      this._timer = setTimeout(loop, Math.max(6, this.cfg.dt * 1000 / this.speed));
    };
    this._timer = setTimeout(loop, Math.max(6, this.cfg.dt * 1000 / this.speed));
  },

  pause() {
    this.playing = false;
    if (this._timer) clearTimeout(this._timer);
  },

  setSpeed(s) {
    const was = this.playing;
    this.pause();
    this.speed = s;
    if (was) this.play();
  },
};
