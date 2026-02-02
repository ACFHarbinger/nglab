import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { 
  Database, 
  Trash2, 
  ExternalLink, 
  FileText, 
  Search, 
  Clock, 
  HardDrive,
  Info
} from "lucide-react";
import clsx from "clsx";

/**
 * Metadata for a historical dataset.
 */
export interface DatasetInfo {
  name: string;
  path: string;
  size: number;
  last_modified: number;
}

/**
 * Component for listing and managing historical datasets (.csv files).
 * Provides search, preview, and deletion capabilities.
 */
export function DataCatalog() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [status, setStatus] = useState<string | null>(null);

  /**
   * Fetches the list of datasets from the backend.
   */
  const fetchDatasets = async () => {
    setLoading(true);
    try {
      const data = await invoke<DatasetInfo[]>("list_datasets");
      setDatasets(data);
    } catch (err) {
      console.error("Failed to fetch datasets:", err);
      setStatus(`Error: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDatasets();
  }, []);

  /**
   * Deletes a dataset after user confirmation.
   * @param path The filesystem path of the dataset to delete.
   */
  const handleDelete = async (path: string, name: string) => {
    if (!confirm(`Are you sure you want to delete ${name}?`)) return;

    try {
      await invoke("delete_dataset", { path });
      setStatus(`Deleted ${name}`);
      fetchDatasets();
    } catch (err) {
      console.error("Failed to delete dataset:", err);
      setStatus(`Error: ${err}`);
    }
  };

  /**
   * Previews the columns of a dataset using existing inference commands.
   * @param path The filesystem path of the dataset.
   */
  const handlePreview = async (path: string) => {
    try {
      const columns = await invoke<string[]>("list_csv_columns", { csvPath: path });
      alert(`Columns in ${path.split('/').pop()}:\n\n${columns.join(", ")}`);
    } catch (err) {
      alert(`Failed to preview columns: ${err}`);
    }
  };

  const filteredDatasets = datasets.filter(ds => 
    ds.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatDate = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleString();
  };

  return (
    <div className="space-y-4">
      {/* Search and Stats */}
      <div className="flex flex-col md:flex-row gap-4 justify-between items-start md:items-center">
        <div className="relative w-full md:w-96">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
          <input
            type="text"
            placeholder="Search datasets..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-slate-900 border border-slate-800 rounded-lg pl-10 pr-4 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          />
        </div>
        
        <div className="flex gap-4">
            <div className="flex items-center gap-2 text-xs text-slate-400">
                <Database className="w-3.5 h-3.5" />
                <span>{datasets.length} Datasets</span>
            </div>
            <div className="flex items-center gap-2 text-xs text-slate-400">
                <HardDrive className="w-3.5 h-3.5" />
                <span>{formatSize(datasets.reduce((sum, d) => sum + d.size, 0))} Total</span>
            </div>
        </div>
      </div>

      {/* Catalog Table */}
      <div className="bg-slate-900/40 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="bg-slate-800/50 border-b border-slate-800">
              <th className="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Name</th>
              <th className="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Size</th>
              <th className="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Last Modified</th>
              <th className="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wider text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/50">
            {loading ? (
              <tr>
                <td colSpan={4} className="px-4 py-12 text-center text-slate-500">
                  <div className="flex flex-col items-center gap-2">
                    <div className="w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
                    <span>Loading catalog...</span>
                  </div>
                </td>
              </tr>
            ) : filteredDatasets.length === 0 ? (
              <tr>
                <td colSpan={4} className="px-4 py-12 text-center text-slate-500">
                  <div className="flex flex-col items-center gap-2">
                    <Info className="w-8 h-8 opacity-20" />
                    <span>{searchQuery ? "No matching datasets found." : "No datasets available. Start by scraping some data!"}</span>
                  </div>
                </td>
              </tr>
            ) : (
              filteredDatasets.map((ds) => (
                <tr key={ds.path} className="group hover:bg-slate-800/30 transition-colors">
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-3">
                      <div className="p-2 bg-indigo-500/10 rounded-lg text-indigo-400 group-hover:bg-indigo-500/20 transition-colors text-xs font-mono">
                         CSV
                      </div>
                      <div className="flex flex-col">
                        <span className="text-sm font-medium text-slate-200 truncate max-w-[300px]" title={ds.name}>
                          {ds.name}
                        </span>
                        <span className="text-[10px] text-slate-500 truncate max-w-[300px]">
                          {ds.path}
                        </span>
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <span className="text-sm text-slate-400">{formatSize(ds.size)}</span>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex flex-col">
                        <span className="text-sm text-slate-300">{formatDate(ds.last_modified)}</span>
                        <div className="flex items-center gap-1 text-[10px] text-slate-500">
                           <Clock className="w-2.5 h-2.5" />
                           <span>Historical</span>
                        </div>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-right">
                    <div className="flex justify-end gap-2">
                      <button
                        onClick={() => handlePreview(ds.path)}
                        title="Preview Columns"
                        className="p-1.5 hover:bg-slate-700 rounded-md text-slate-400 hover:text-indigo-400 transition-colors"
                      >
                        <ExternalLink className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => handleDelete(ds.path, ds.name)}
                        title="Delete Dataset"
                        className="p-1.5 hover:bg-rose-900/30 rounded-md text-slate-400 hover:text-rose-400 transition-colors"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {status && (
        <div className={clsx(
          "px-4 py-2 rounded-lg text-sm",
          status.includes("Error") ? "bg-rose-900/20 text-rose-400 border border-rose-900/50" : "bg-emerald-900/20 text-emerald-400 border border-emerald-900/50"
        )}>
          {status}
        </div>
      )}
    </div>
  );
}
