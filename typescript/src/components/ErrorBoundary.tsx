import React, { Component, ErrorInfo, ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, errorInfo: null };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught error:", error, errorInfo);
    this.setState({ error, errorInfo });
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="p-8 bg-slate-950 text-white h-screen overflow-auto">
          <h1 className="text-2xl text-rose-500 font-bold mb-4">
            Something went wrong.
          </h1>
          <div className="bg-slate-900 p-4 rounded border border-slate-800 font-mono text-sm whitespace-pre-wrap">
            <p className="text-rose-300 font-bold mb-2">
              {this.state.error?.toString()}
            </p>
            <hr className="border-slate-700 my-2" />
            <p className="text-slate-400">
              {this.state.errorInfo?.componentStack}
            </p>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
