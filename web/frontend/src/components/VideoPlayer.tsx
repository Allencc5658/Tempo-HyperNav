/**
 * Synchronized video player component.
 * Supports external step control for timeline synchronization with graph viewer.
 */

import { useRef, useEffect, useCallback } from "react";

interface Props {
  src: string;
  totalSteps: number;
  currentStep: number;
  onStepChange?: (step: number) => void;
  fps?: number;
}

export default function VideoPlayer({
  src,
  totalSteps,
  currentStep,
  onStepChange,
  fps = 30,
}: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);

  // Sync video position when currentStep changes externally
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !video.duration || totalSteps <= 0) return;
    const targetTime = (currentStep / totalSteps) * video.duration;
    // Only seek if the difference is significant (avoid feedback loops)
    if (Math.abs(video.currentTime - targetTime) > 0.5) {
      video.currentTime = targetTime;
    }
  }, [currentStep, totalSteps]);

  // Report step changes when video plays
  const handleTimeUpdate = useCallback(() => {
    const video = videoRef.current;
    if (!video || !onStepChange || totalSteps <= 0) return;
    const step = Math.round((video.currentTime / video.duration) * totalSteps);
    onStepChange(step);
  }, [onStepChange, totalSteps]);

  return (
    <video
      ref={videoRef}
      src={src}
      controls
      onTimeUpdate={handleTimeUpdate}
      className="w-full rounded-lg"
      preload="metadata"
    />
  );
}
