"use client";

import { useState, useRef, useEffect, useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import dynamic from "next/dynamic";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { SendIcon, Brain, BookOpen, Database, Sparkles, Loader2, Map as MapIcon, X, Trash2, TerminalSquare, User } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import("react-plotly.js"), { 
  ssr: false, 
  loading: () => (
    <div className="flex-1 flex flex-col items-center justify-center space-y-4">
      <Loader2 className="h-8 w-8 animate-spin text-zinc-600"/>
      <div className="text-zinc-500 text-sm font-mono animate-pulse">Initializing 3D renderer...</div>
    </div>
  ) 
});

type ContextDoc = { content: string; source: string };

type Message = { 
  role: "user" | "assistant"; 
  content: string;
  contexts?: ContextDoc[];
};

type VectorPoint = {
  id: number;
  x: number;
  y: number;
  z: number;
  text: string;
  source: string;
};

const INITIAL_MESSAGE: Message = { 
  role: "assistant", 
  content: "Welcome to your General Relativity environment. I am tuned to provide rigorous mathematical formulations. Ask me about tensors, manifolds, or Einstein's field equations." 
};

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([INITIAL_MESSAGE]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  
  // LocalStorage initialization
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    const saved = localStorage.getItem("relav_messages");
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (parsed.length > 0) setMessages(parsed);
      } catch (e) {
        console.error("Failed to parse saved messages");
      }
    }
    setIsLoaded(true);
  }, []);

  useEffect(() => {
    if (isLoaded) {
      localStorage.setItem("relav_messages", JSON.stringify(messages));
    }
  }, [messages, isLoaded]);

  // RAG Visualizer State
  const [activeContexts, setActiveContexts] = useState<ContextDoc[]>([]);
  const [retrieving, setRetrieving] = useState(false);

  // Vector Space Map State
  const [showVectorMap, setShowVectorMap] = useState(false);
  const [vectorData, setVectorData] = useState<VectorPoint[]>([]);
  const [loadingVector, setLoadingVector] = useState(false);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const fetchVectorSpace = async () => {
    setShowVectorMap(true);
    if (vectorData.length > 0) return;
    
    setLoadingVector(true);
    try {
      const res = await fetch("http://localhost:8000/vector-space");
      if (!res.ok) throw new Error("Failed to fetch vector space");
      const data = await res.json();
      setVectorData(data.points || []);
    } catch (err) {
      console.error(err);
    } finally {
      setLoadingVector(false);
    }
  };

  const clearChat = () => {
    if (confirm("Are you sure you want to clear the conversation?")) {
      setMessages([INITIAL_MESSAGE]);
      setActiveContexts([]);
      localStorage.removeItem("relav_messages");
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { role: "user" as const, content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    setActiveContexts([]);
    setRetrieving(true);
    setShowVectorMap(false);

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage.content }),
      });

      if (!response.ok) throw new Error("Network response was not ok");
      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantContent = "";
      
      setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split("\n\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const dataStr = line.substring(6);
            if (dataStr === "[DONE]") break;

            try {
              const data = JSON.parse(dataStr);
              if (data.type === "context") {
                setRetrieving(false);
                setActiveContexts(data.data);
                setMessages((prev) => {
                  const newMsgs = [...prev];
                  newMsgs[newMsgs.length - 1].contexts = data.data;
                  return newMsgs;
                });
              } else if (data.type === "token") {
                setRetrieving(false);
                assistantContent += data.data;
                setMessages((prev) => {
                  const newMsgs = [...prev];
                  newMsgs[newMsgs.length - 1].content = assistantContent;
                  return newMsgs;
                });
              }
            } catch (e) {
              console.error("Error parsing stream chunk", e);
            }
          }
        }
      }
    } catch (error) {
      console.error("Error fetching chat:", error);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Error: Connection to the RAG backend failed. Please ensure the FastAPI server is running." },
      ]);
    } finally {
      setIsLoading(false);
      setRetrieving(false);
    }
  };

  const getColorForSource = (source: string) => {
    let hash = 0;
    for (let i = 0; i < source.length; i++) {
      hash = source.charCodeAt(i) + ((hash << 5) - hash);
    }
    const hue = Math.abs(hash % 360);
    return `hsl(${hue}, 70%, 60%)`;
  };

  // Prepare Plotly 3D Data
  const plotData = useMemo(() => {
    if (!vectorData.length) return [];
    
    const sources = Array.from(new Set(vectorData.map(d => d.source)));
    return sources.map(source => {
      const sourceData = vectorData.filter(d => d.source === source);
      return {
        x: sourceData.map(d => d.x),
        y: sourceData.map(d => d.y),
        z: sourceData.map(d => d.z),
        text: sourceData.map(d => {
          return `<b>${source}</b><br>${d.text.replace(/.{50}/g, '$&<br>')}`;
        }),
        mode: "markers",
        type: "scatter3d",
        name: source.length > 20 ? source.substring(0, 20) + "..." : source,
        hoverinfo: "text",
        marker: {
          size: 3, // Smaller points to reduce overlap
          color: getColorForSource(source),
          opacity: 0.35, // Less alpha
          line: { width: 0 }
        }
      } as any;
    });
  }, [vectorData]);

  if (!isLoaded) return null;

  return (
    <div className="flex h-screen w-full bg-[#050505] font-sans text-zinc-100 selection:bg-zinc-800 overflow-hidden">
      
      {/* Main Content Area */}
      <div className="flex-1 flex flex-col h-full relative">
        
        {/* Top Navbar */}
        <div className="absolute top-0 w-full h-16 border-b border-white/5 bg-[#050505]/80 backdrop-blur-xl z-50 flex items-center justify-between px-6">
          <div className="flex items-center gap-3">
            <div className="text-zinc-100">
              <Brain className="h-5 w-5" />
            </div>
            <h1 className="text-sm font-semibold tracking-wide text-zinc-200">relativity_agent <span className="text-zinc-600 font-normal">v1.0</span></h1>
          </div>
          
          <div className="flex items-center gap-4">
            <Button 
              variant="ghost" 
              size="sm"
              onClick={clearChat}
              className="text-zinc-500 hover:text-zinc-200 hover:bg-white/5 transition-colors h-8"
              title="Clear Memory"
            >
              <Trash2 className="h-4 w-4 mr-2" /> Clear
            </Button>
            <div className="h-4 w-px bg-white/10"></div>
            <Button 
              variant="ghost" 
              size="sm" 
              onClick={showVectorMap ? () => setShowVectorMap(false) : fetchVectorSpace}
              className={`h-8 transition-colors ${showVectorMap ? 'bg-zinc-100 text-black hover:bg-zinc-200' : 'text-zinc-400 hover:text-zinc-100 hover:bg-white/5'}`}
            >
              {showVectorMap ? (
                <><X className="h-4 w-4 mr-2" /> Exit Map</>
              ) : (
                <><MapIcon className="h-4 w-4 mr-2" /> 3D Topology</>
              )}
            </Button>
          </div>
        </div>

        {/* Dynamic View: Map vs Chat */}
        {showVectorMap ? (
          <div className="flex-1 w-full h-full pt-16 relative bg-[#0a0a0a]">
            <div className="absolute top-24 left-8 z-40 pointer-events-none">
              <h2 className="text-2xl font-light text-zinc-100 tracking-tight">Latent Space</h2>
              <p className="text-xs text-zinc-500 font-mono mt-2">PCA Reduction (n=3) of Mistral Embeddings</p>
            </div>
            
            {loadingVector ? (
              <div className="flex-1 h-full flex flex-col items-center justify-center space-y-4">
                <Loader2 className="h-8 w-8 text-zinc-600 animate-spin" />
                <span className="text-zinc-500 font-mono text-xs animate-pulse">Computing manifold...</span>
              </div>
            ) : vectorData.length > 0 ? (
              <div className="w-full h-full">
                <Plot
                  data={plotData}
                  layout={{
                    autosize: true,
                    margin: { l: 0, r: 0, b: 0, t: 0 },
                    paper_bgcolor: "transparent",
                    plot_bgcolor: "transparent",
                    font: { color: "#71717a", family: "inherit", size: 10 },
                    scene: {
                      xaxis: { visible: false, showgrid: false, zeroline: false },
                      yaxis: { visible: false, showgrid: false, zeroline: false },
                      zaxis: { visible: false, showgrid: false, zeroline: false },
                      camera: { eye: { x: 1.2, y: 1.2, z: 1.2 } }
                    },
                    showlegend: true,
                    legend: { 
                      font: { color: "#71717a", size: 10 },
                      bgcolor: "rgba(0,0,0,0)",
                      itemsizing: "constant",
                      x: 0.02,
                      y: 0.9,
                    },
                    hoverlabel: {
                      bgcolor: "#18181b",
                      bordercolor: "#27272a",
                      font: { color: "#f4f4f5", size: 11, family: "inherit" }
                    }
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: "100%", height: "100%" }}
                />
              </div>
            ) : (
              <div className="flex-1 h-full flex items-center justify-center text-zinc-600 font-mono text-sm">
                [ No vectors found in ChromaDB ]
              </div>
            )}
          </div>
        ) : (
          <div className="flex-1 flex flex-col pt-16 relative min-h-0">
            {/* Normal Chat View - Centered and clean */}
            <div 
              ref={scrollRef}
              className="flex-1 overflow-y-auto scroll-smooth pb-40"
            >
              <div className="max-w-4xl mx-auto w-full flex flex-col space-y-8 px-6 py-12">
                {messages.map((m, i) => (
                  <motion.div 
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    key={i} 
                    className="flex gap-6 w-full group"
                  >
                    <div className="shrink-0 mt-1">
                      {m.role === "assistant" ? (
                        <div className="w-8 h-8 rounded bg-white/10 border border-white/10 flex items-center justify-center shadow-sm">
                          <TerminalSquare className="w-4 h-4 text-zinc-300" />
                        </div>
                      ) : (
                        <div className="w-8 h-8 rounded bg-zinc-800 border border-zinc-700 flex items-center justify-center">
                          <User className="w-4 h-4 text-zinc-400" />
                        </div>
                      )}
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="font-medium text-xs text-zinc-500 mb-2 tracking-wide uppercase">
                        {m.role === "assistant" ? "System" : "User"}
                      </div>
                      
                      {m.role === "user" ? (
                        <p className="text-[17px] leading-relaxed text-zinc-200 whitespace-pre-wrap font-light">{m.content}</p>
                      ) : (
                        <div className="prose prose-invert prose-zinc max-w-none text-[17px] prose-p:leading-relaxed prose-p:text-[17px] prose-p:font-light prose-pre:bg-[#0a0a0a] prose-pre:border prose-pre:border-white/5 prose-math:text-zinc-100 prose-li:text-[17px] prose-headings:text-zinc-100">
                          {m.content.length > 0 ? (
                            <ReactMarkdown
                              remarkPlugins={[remarkMath]}
                              rehypePlugins={[rehypeKatex]}
                            >
                              {m.content}
                            </ReactMarkdown>
                          ) : (
                            <div className="flex items-center gap-3 text-zinc-500 h-6">
                              <span className="relative flex h-2 w-2">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-zinc-400 opacity-50"></span>
                                <span className="relative inline-flex rounded-full h-2 w-2 bg-zinc-500"></span>
                              </span>
                              <span className="text-xs font-mono animate-pulse">computing...</span>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Input Area Overlay */}
            <div className="absolute bottom-0 w-full bg-gradient-to-t from-[#050505] via-[#050505] to-transparent pt-20 pb-8 px-6">
              <div className="max-w-4xl mx-auto relative">
                <form onSubmit={handleSubmit} className="relative flex items-center">
                  <div className="absolute left-4 text-zinc-500">
                    <Sparkles className="h-5 w-5" />
                  </div>
                  <Input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Enter mathematical query..."
                    className="w-full pl-12 pr-16 py-7 bg-[#0a0a0a] border-white/10 rounded-2xl text-zinc-100 placeholder:text-zinc-600 focus-visible:ring-1 focus-visible:ring-white/20 text-base shadow-2xl transition-all"
                    disabled={isLoading}
                  />
                  <Button 
                    type="submit" 
                    disabled={isLoading || !input.trim()}
                    size="icon"
                    className="absolute right-3 w-10 h-10 rounded-xl bg-white text-black hover:bg-zinc-200 transition-all disabled:opacity-50 disabled:bg-zinc-800 disabled:text-zinc-500"
                  >
                    {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <SendIcon className="h-4 w-4 ml-0.5" />}
                  </Button>
                </form>
                <div className="text-center mt-3 text-[10px] text-zinc-600 font-mono tracking-widest uppercase">
                  Generative RAG Pipeline Active
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Sleek Right Sidebar for RAG Context */}
      <div className="w-[320px] border-l border-white/5 bg-[#080808] hidden lg:flex flex-col z-20">
        <div className="h-16 border-b border-white/5 flex items-center px-6 gap-3 shrink-0">
          <Database className="h-4 w-4 text-zinc-400" />
          <h2 className="text-xs font-semibold text-zinc-300 tracking-widest uppercase">Active Context</h2>
        </div>
        
        <ScrollArea className="flex-1 px-4 py-6">
          <AnimatePresence>
            {retrieving && (
              <motion.div 
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="mb-6 rounded-xl border border-zinc-800 bg-[#0a0a0a] p-4 text-center overflow-hidden relative"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -translate-x-full animate-[shimmer_1.5s_infinite]"></div>
                <Database className="h-6 w-6 text-zinc-500 mx-auto mb-3 animate-pulse" />
                <p className="text-xs font-mono text-zinc-400">Scanning Manifolds...</p>
              </motion.div>
            )}
          </AnimatePresence>

          <div className="space-y-4">
            {activeContexts.length > 0 && !showVectorMap ? (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
                {activeContexts.map((ctx, idx) => (
                  <motion.div 
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.05 }}
                    key={idx}
                    className="group"
                  >
                    <div className="rounded-xl border border-white/5 bg-[#0a0a0a] p-4 hover:border-white/10 transition-all">
                      <div className="flex items-center gap-2 mb-3">
                        <BookOpen className="h-3 w-3 text-zinc-500" />
                        <span className="font-mono text-[10px] text-zinc-400 truncate tracking-wide">
                          {ctx.source}
                        </span>
                      </div>
                      <p className="text-xs text-zinc-500 leading-relaxed line-clamp-6 font-light">
                        {ctx.content}
                      </p>
                    </div>
                  </motion.div>
                ))}
              </motion.div>
            ) : !retrieving && messages.length > 1 && !showVectorMap ? (
              <div className="flex flex-col items-center justify-center h-32 text-zinc-700">
                <p className="text-xs font-mono">No relevant context found.</p>
              </div>
            ) : showVectorMap ? (
              <div className="text-xs text-zinc-500 leading-relaxed bg-[#0a0a0a] p-4 rounded-xl border border-white/5">
                <span className="text-zinc-300 font-medium block mb-2 uppercase tracking-widest text-[10px]">3D Mode Active</span>
                The knowledge graph displays embeddings mapped via PCA. Closer points represent semantic similarity in physical concepts.
              </div>
            ) : null}
          </div>
        </ScrollArea>
      </div>
    </div>
  );
}
