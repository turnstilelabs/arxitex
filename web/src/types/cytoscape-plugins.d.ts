declare module "react-cytoscapejs";
declare module "cytoscape-dagre";

declare global {
    interface Window {
        MathJax?: {
            typesetPromise?: (elements?: any[]) => Promise<void>;
        };
    }
}

export { };
