// Messages between the page and the model worker (src/worker.ts).

export interface InitMessage {
  type: "init";
  /** Absolute URL of the folder that holds models/ and wasm/, without a trailing slash. */
  base: string;
  /** Weight of the newest segmenter mask in the running average (CONFIG mask_ema_baru). */
  maskEma: number;
}

export interface FrameMessage {
  type: "frame";
  id: number;
  /** Capture time in ms (performance.now()). */
  ts: number;
  /** The camera frame as captured (not mirrored); it comes back with the result. */
  bitmap: ImageBitmap;
}

/** Forget the mask average, so the next mask starts it over (after a jump, like the Python reset). */
export interface ResetMessage {
  type: "reset";
}

export type ToWorker = InitMessage | FrameMessage | ResetMessage;

export interface Hand {
  /** 21 landmarks as x0, y0, x1, y1... normalized 0..1 in the mirrored picture. */
  landmarks: Float32Array;
  /** Top GestureRecognizer category ("None" when there is none) and its score. */
  gesture: string;
  score: number;
}

export interface ProgressMessage {
  type: "progress";
  loaded: number;
  total: number;
}

export interface ReadyMessage {
  type: "ready";
  /** "GPU" or "CPU", the delegate the models run on. */
  delegate: string;
}

export interface ErrorMessage {
  type: "error";
  message: string;
}

export interface ResultMessage {
  type: "result";
  id: number;
  ts: number;
  bitmap: ImageBitmap;
  hands: Hand[];
  /** Person confidence 0..1 at maskWidth x maskHeight (half the frame), mirrored, averaged over frames. */
  mask: Float32Array;
  maskWidth: number;
  maskHeight: number;
  handMs: number;
  segmenterMs: number;
}

export type FromWorker = ProgressMessage | ReadyMessage | ErrorMessage | ResultMessage;
