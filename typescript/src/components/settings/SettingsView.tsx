import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { 
  Shield, 
  Key, 
  Save, 
  Trash2, 
  Eye, 
  EyeOff, 
  CheckCircle, 
  AlertCircle,
  RefreshCw
} from "lucide-react";
import clsx from "clsx";

interface Integration {
  id: number;
  service_name: string;
  created_at: string;
}

export function SettingsView() {
  const [integrations, setIntegrations] = useState<Integration[]>([]);
  const [loading, setLoading] = useState(false);
  const [saveLoading, setSaveLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Form states
  const [selectedExchange, setSelectedExchange] = useState("Binance");
  const [apiKey, setApiKey] = useState("");
  const [secret, setSecret] = useState("");
  const [showSecret, setShowSecret] = useState(false);

  useEffect(() => {
    fetchIntegrations();
  }, []);

  const fetchIntegrations = async () => {
    setLoading(true);
    try {
      const resp: any = await invoke("list_integrations");
      if (resp.success) {
        setIntegrations(resp.data || []);
      } else {
        setError(resp.message);
      }
    } catch (e) {
      console.error("Failed to fetch integrations", e);
      setError("Vault is locked or service is unavailable");
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    if (!apiKey || !secret) {
      setError("API Key and Secret are required");
      return;
    }

    setSaveLoading(selectedExchange);
    setError(null);
    setSuccess(null);

    try {
      const resp: any = await invoke("save_exchange_integration", {
        exchange: selectedExchange,
        apiKey,
        secret
      });

      if (resp.success) {
        setSuccess(`${selectedExchange} API credentials saved securely.`);
        setApiKey("");
        setSecret("");
        fetchIntegrations();
      } else {
        setError(resp.message);
      }
    } catch (e) {
      console.error("Failed to save integration", e);
      setError("An unexpected error occurred. Is the vault unlocked?");
    } finally {
      setSaveLoading(null);
    }
  };

  const handleDelete = async (id: number) => {
    if (!confirm("Are you sure you want to remove these credentials?")) return;

    try {
      const resp: any = await invoke("delete_integration", { id });
      if (resp.success) {
        fetchIntegrations();
      } else {
        setError(resp.message);
      }
    } catch (e) {
      console.error("Failed to delete integration", e);
    }
  };

  return (
    <div className="p-8 max-w-4xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="flex items-center gap-3 border-b border-slate-800 pb-6">
        <div className="p-3 bg-indigo-500/10 rounded-xl text-indigo-400">
          <Shield size={32} />
        </div>
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight">Security & Integrations</h1>
          <p className="text-slate-400">Manage your encrypted API keys and external exchange connections.</p>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-rose-500/10 border border-rose-500/20 rounded-xl text-rose-400 flex items-center gap-3 animate-in zoom-in-95">
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      {success && (
        <div className="p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-xl text-emerald-400 flex items-center gap-3 animate-in zoom-in-95">
          <CheckCircle size={20} />
          <span>{success}</span>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* API Key Form */}
        <section className="bg-slate-900/50 rounded-2xl border border-slate-800 p-6 space-y-6">
          <div className="flex items-center gap-2 text-white font-semibold">
            <Key size={20} className="text-indigo-400" />
            <h2>Add New Credentials</h2>
          </div>

          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-xs uppercase tracking-widest text-slate-500 font-bold">Exchange</label>
              <select 
                value={selectedExchange}
                onChange={(e) => setSelectedExchange(e.target.value)}
                className="w-full bg-slate-950 border border-slate-700 rounded-lg px-4 py-2 text-white focus:ring-2 focus:ring-indigo-500 outline-none transition-all"
              >
                <option>Binance</option>
                <option>Kraken</option>
                <option>Deribit</option>
              </select>
            </div>

            <div className="space-y-2">
              <label className="text-xs uppercase tracking-widest text-slate-500 font-bold">API Key</label>
              <input 
                type="text" 
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="Enter API Key"
                className="w-full bg-slate-950 border border-slate-700 rounded-lg px-4 py-2 text-white focus:ring-2 focus:ring-indigo-500 outline-none transition-all placeholder:text-slate-700"
              />
            </div>

            <div className="space-y-2 relative">
              <label className="text-xs uppercase tracking-widest text-slate-500 font-bold">API Secret</label>
              <div className="relative">
                <input 
                  type={showSecret ? "text" : "password"} 
                  value={secret}
                  onChange={(e) => setSecret(e.target.value)}
                  placeholder="Enter API Secret"
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-4 py-2 pr-12 text-white focus:ring-2 focus:ring-indigo-500 outline-none transition-all placeholder:text-slate-700"
                />
                <button 
                  onClick={() => setShowSecret(!showSecret)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-white transition-colors"
                >
                  {showSecret ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>
            </div>

            <button 
              onClick={handleSave}
              disabled={!!saveLoading}
              className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-800 disabled:text-slate-600 px-4 py-3 rounded-lg text-white font-bold transition-all shadow-lg shadow-indigo-600/20"
            >
              {saveLoading ? <RefreshCw className="animate-spin" size={20} /> : <Save size={20} />}
              <span>{saveLoading ? "Saving..." : "Save Securely"}</span>
            </button>
          </div>
        </section>

        {/* Existing Integrations */}
        <section className="bg-slate-900/50 rounded-2xl border border-slate-800 p-6 flex flex-col">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-2 text-white font-semibold">
              <CheckCircle size={20} className="text-emerald-400" />
              <h2>Active Connections</h2>
            </div>
            <button 
              onClick={fetchIntegrations}
              className="p-2 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-white transition-all"
            >
              <RefreshCw size={16} className={clsx(loading && "animate-spin")} />
            </button>
          </div>

          <div className="flex-1 space-y-3">
            {integrations.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center p-8 space-y-3 opacity-50">
                <Shield size={48} className="text-slate-700" />
                <p className="text-sm text-slate-500">No active integrations found.<br/>Unlock your vault to view secrets.</p>
              </div>
            ) : (
              integrations.map((int) => (
                <div 
                  key={int.id}
                  className="flex items-center justify-between p-4 bg-slate-950 rounded-xl border border-slate-800 group hover:border-indigo-500/50 transition-all"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-indigo-500/10 flex items-center justify-center text-indigo-400 font-bold uppercase">
                      {int.service_name.charAt(0)}
                    </div>
                    <div>
                      <h3 className="text-white font-bold capitalize">{int.service_name}</h3>
                      <p className="text-[10px] text-slate-500 uppercase tracking-tighter">Connected {new Date(int.created_at).toLocaleDateString()}</p>
                    </div>
                  </div>
                  <button 
                    onClick={() => handleDelete(int.id)}
                    className="p-2 text-slate-600 hover:text-rose-500 hover:bg-rose-500/10 rounded-lg opacity-0 group-hover:opacity-100 transition-all"
                  >
                    <Trash2 size={18} />
                  </button>
                </div>
              ))
            )}
            {integrations.length > 0 && (
              <button 
                onClick={async () => {
                  setLoading(true);
                  try {
                    await invoke("reconnect_exchanges");
                    setSuccess("Exchanges reconnected successfully.");
                  } catch (e) {
                    setError("Failed to reconnect. Is the vault unlocked?");
                  } finally {
                    setLoading(false);
                  }
                }}
                className="mt-4 w-full flex items-center justify-center gap-2 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white px-4 py-2 rounded-lg transition-all"
              >
                <RefreshCw size={16} className={clsx(loading && "animate-spin")} />
                <span>Reconnect All Exchanges</span>
              </button>
            )}
          </div>

          <div className="mt-6 p-4 bg-indigo-500/5 rounded-xl border border-indigo-500/10">
            <p className="text-[11px] text-slate-400 leading-relaxed">
              <span className="text-indigo-400 font-bold mr-1">Note:</span>
              Your keys are encrypted using AES-256 and stored in a secure local vault. They never leave your device unencrypted.
            </p>
          </div>
        </section>
      </div>
    </div>
  );
}
