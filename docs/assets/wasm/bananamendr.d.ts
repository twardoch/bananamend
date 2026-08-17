/* tslint:disable */
/* eslint-disable */

/**
 * A loaded BananaMind-2 checkpoint.
 */
export class Model {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Applies the chat format of the checkpoint and gives the text.
     */
    applyChatTemplate(messages: any): string;
    /**
     * Applies the chat format, and then writes the answer of the assistant.
     */
    chat(messages: any, options: any, on_token?: Function | null): any;
    /**
     * Applies the chat format and gives the token ids.
     */
    chatTokens(messages: any): Uint32Array;
    /**
     * Gives the text of token ids.
     */
    detokenize(tokens: Uint32Array, skip_special_tokens: boolean): string;
    /**
     * Builds a model from the four parts of a checkpoint.
     *
     * Give the text of `config.json`, the text of `tokenizer.json`, the text of
     * `tokenizer_config.json` (or nothing), and the bytes of
     * `model.safetensors`.
     */
    static fromParts(config_json: string, tokenizer_json: string, tokenizer_config_json: string | null | undefined, weights: Uint8Array): Model;
    /**
     * Continues a text. The chat format is not applied.
     *
     * `on_token` is optional. If you give a function, this module calls it for
     * each new token with the text of that token and its id.
     */
    generate(prompt: string, options: any, on_token?: Function | null): any;
    /**
     * Gives the architecture and tokenizer facts of the checkpoint.
     */
    info(): any;
    /**
     * Gives the scores of all of the next possible tokens for a text.
     *
     * The text goes to the tokenizer without a change. No token is added, which
     * is what the Python module does as well.
     */
    logits(text: string): Float32Array;
    /**
     * Gives the token ids of a text.
     */
    tokenize(text: string): Uint32Array;
}

/**
 * Gives the version of the crate that built this module.
 */
export function version(): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_model_free: (a: number, b: number) => void;
    readonly model_applyChatTemplate: (a: number, b: any) => [number, number, number, number];
    readonly model_chat: (a: number, b: any, c: any, d: number) => [number, number, number];
    readonly model_chatTokens: (a: number, b: any) => [number, number, number, number];
    readonly model_detokenize: (a: number, b: number, c: number, d: number) => [number, number, number, number];
    readonly model_fromParts: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number];
    readonly model_generate: (a: number, b: number, c: number, d: any, e: number) => [number, number, number];
    readonly model_info: (a: number) => [number, number, number];
    readonly model_logits: (a: number, b: number, c: number) => [number, number, number, number];
    readonly model_tokenize: (a: number, b: number, c: number) => [number, number, number, number];
    readonly version: () => [number, number];
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
