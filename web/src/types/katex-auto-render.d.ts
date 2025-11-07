declare module "katex/contrib/auto-render" {
    type Options = {
        delimiters?: { left: string; right: string; display: boolean }[];
        throwOnError?: boolean;
        errorColor?: string;
        strict?: "ignore" | "warn" | "error" | ((errorCode: string, errorMsg: string, token?: any) => "ignore" | "warn" | "error");
        trust?: boolean | ((context: { command: string }) => boolean);
    };
    const renderMathInElement: (el: HTMLElement, options?: Options) => void;
    export default renderMathInElement;
}
declare module "katex/dist/contrib/auto-render.mjs" {
    type Options = {
        delimiters?: { left: string; right: string; display: boolean }[];
        throwOnError?: boolean;
        errorColor?: string;
        strict?: "ignore" | "warn" | "error" | ((errorCode: string, errorMsg: string, token?: any) => "ignore" | "warn" | "error");
        trust?: boolean | ((context: { command: string }) => boolean);
    };
    const renderMathInElement: (el: HTMLElement, options?: Options) => void;
    export default renderMathInElement;
}
