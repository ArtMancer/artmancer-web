"use client";

import { useServer } from "@/contexts/ServerContext";
import { MdCloudOff, MdCloudDone, MdRefresh } from "react-icons/md";
import { useState, useRef, useEffect } from "react";

export default function ServerControl() {
  const { status, serviceStatus, checkStatus } = useServer();
  const [showStatusTooltip, setShowStatusTooltip] = useState(false);
  const [showRefreshTooltip, setShowRefreshTooltip] = useState(false);
  const statusTooltipTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const refreshTooltipTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const [isChecking, setIsChecking] = useState(false);

  const getStatusInfo = () => {
    switch (status) {
      case "online":
        return {
          icon: MdCloudDone,
          color: "text-green-500 dark:text-green-400",
          bgColor: "bg-green-500/10 dark:bg-green-500/20",
          hoverBg: "hover:bg-green-500/20 dark:hover:bg-green-500/30",
          tooltip: "Server is online and ready",
          label: "Online",
        };
      default:
        return {
          icon: MdCloudOff,
          color: "text-red-500 dark:text-red-400",
          bgColor: "bg-red-500/10 dark:bg-red-500/20",
          hoverBg: "hover:bg-red-500/20 dark:hover:bg-red-500/30",
          tooltip: "Server is offline",
          label: "Offline",
        };
    }
  };

  const statusInfo = getStatusInfo();
  const StatusIcon = statusInfo.icon;

  const handleManualCheck = async () => {
    if (isChecking) return;
    setIsChecking(true);
    try {
      await checkStatus(true); // true = hiện notification
    } finally {
      setIsChecking(false);
    }
  };

  // Cleanup timeouts on unmount
  useEffect(() => {
    return () => {
      if (statusTooltipTimeoutRef.current)
        clearTimeout(statusTooltipTimeoutRef.current);
      if (refreshTooltipTimeoutRef.current)
        clearTimeout(refreshTooltipTimeoutRef.current);
    };
  }, []);

  const handleStatusMouseEnter = () => {
    statusTooltipTimeoutRef.current = setTimeout(() => {
      setShowStatusTooltip(true);
    }, 750);
  };

  const handleStatusMouseLeave = () => {
    if (statusTooltipTimeoutRef.current) {
      clearTimeout(statusTooltipTimeoutRef.current);
      statusTooltipTimeoutRef.current = null;
    }
    setShowStatusTooltip(false);
  };

  const handleRefreshMouseEnter = () => {
    refreshTooltipTimeoutRef.current = setTimeout(() => {
      setShowRefreshTooltip(true);
    }, 750);
  };

  const handleRefreshMouseLeave = () => {
    if (refreshTooltipTimeoutRef.current) {
      clearTimeout(refreshTooltipTimeoutRef.current);
      refreshTooltipTimeoutRef.current = null;
    }
    setShowRefreshTooltip(false);
  };

  return (
    <div className="flex items-center gap-2">
      {/* Manual Check Button */}
      <div className="relative">
        <button
          onClick={handleManualCheck}
          disabled={isChecking}
          onMouseEnter={handleRefreshMouseEnter}
          onMouseLeave={handleRefreshMouseLeave}
          className="p-3 rounded-lg transition-all duration-200 h-12 w-12 flex items-center justify-center bg-blue-500/10 dark:bg-blue-500/20 hover:bg-blue-500/20 dark:hover:bg-blue-500/30 text-blue-500 dark:text-blue-400 disabled:opacity-50 disabled:cursor-not-allowed"
          aria-label="Check server status"
        >
          <MdRefresh size={20} className={isChecking ? "animate-spin" : ""} />
        </button>
        {showRefreshTooltip && (
          <div className="absolute right-0 top-full mt-2 px-2 py-1 bg-[var(--primary-bg)] border border-[var(--border-color)] rounded-lg shadow-lg text-xs text-[var(--text-primary)] z-50 whitespace-nowrap transition-all duration-200 opacity-100 translate-y-0">
            Check status
          </div>
        )}
      </div>
      {/* Status Indicator */}
      <div className="relative">
        <button
          className={`p-3 rounded-lg transition-all duration-200 ${statusInfo.bgColor} ${statusInfo.hoverBg} ${statusInfo.color} h-12 w-12 flex items-center justify-center`}
          aria-label={`Server status: ${statusInfo.label}`}
          onMouseEnter={handleStatusMouseEnter}
          onMouseLeave={handleStatusMouseLeave}
        >
          <StatusIcon size={20} />
        </button>
        {showStatusTooltip && (
          <div className="absolute right-0 top-full mt-2 px-2 py-1.5 bg-[var(--primary-bg)] border border-[var(--border-color)] rounded-lg shadow-lg text-xs text-[var(--text-primary)] z-50 whitespace-nowrap transition-all duration-200 opacity-100 translate-y-0">
            <div className="font-medium mb-1">{statusInfo.label}</div>
            <div className="flex flex-col gap-0.5 text-[10px] opacity-80">
              {serviceStatus.generation !== undefined && (
                <div className="flex items-center gap-1.5">
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${
                      serviceStatus.generation ? "bg-green-500" : "bg-red-500"
                    }`}
                  />
                  <span>
                    Generation:{" "}
                    {serviceStatus.generation ? "Online" : "Offline"}
                  </span>
                </div>
              )}
              {serviceStatus.segmentation !== undefined && (
                <div className="flex items-center gap-1.5">
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${
                      serviceStatus.segmentation ? "bg-green-500" : "bg-red-500"
                    }`}
                  />
                  <span>
                    Segmentation:{" "}
                    {serviceStatus.segmentation ? "Online" : "Offline"}
                  </span>
                </div>
              )}
              {serviceStatus.image_utils !== undefined && (
                <div className="flex items-center gap-1.5">
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${
                      serviceStatus.image_utils ? "bg-green-500" : "bg-red-500"
                    }`}
                  />
                  <span>
                    Image Utils:{" "}
                    {serviceStatus.image_utils ? "Online" : "Offline"}
                  </span>
                </div>
              )}
              {serviceStatus.job_manager !== undefined && (
                <div className="flex items-center gap-1.5">
                  <span
                    className={`w-1.5 h-1.5 rounded-full ${
                      serviceStatus.job_manager ? "bg-green-500" : "bg-red-500"
                    }`}
                  />
                  <span>
                    Job Manager:{" "}
                    {serviceStatus.job_manager ? "Online" : "Offline"}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
