import { get, post, requestBlob } from "./fetch";
import type { FileInfo, FileUploadParams } from "./types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api";

export const fileApi = {
  uploadFile: async (params: FileUploadParams): Promise<FileInfo> => {
    const formData = new FormData();
    formData.append("file", params.file);

    if (params.session_id) {
      formData.append("session_id", params.session_id);
    }

    return post<FileInfo>("/files", formData);
  },

  getFileInfo: (fileId: string): Promise<FileInfo> => {
    return get<FileInfo>(`/files/${fileId}`);
  },

  downloadFile: async (fileId: string): Promise<Blob> => {
    return requestBlob(`/files/${fileId}/download`);
  },

  getFileDownloadUrl: (fileId: string): string => {
    return `${API_BASE_URL}/files/${fileId}/download`;
  },
};
