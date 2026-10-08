import { useState } from "react";

export default function VideoBanner({ collegeName = "Indus University" }) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);

  // YouTube embed video ID (Indus University campus video or college promo)
  const videoId = "VpZ1qL8kZJ8"; // Campus promo / intro video

  if (isMinimized) {
    return (
      <div className="video-banner-minimized">
        <button
          type="button"
          onClick={() => setIsMinimized(false)}
          className="video-restore-btn"
        >
          🎬 Show Campus Video Tour
        </button>
      </div>
    );
  }

  return (
    <div className="video-banner-card">
      <button
        type="button"
        className="video-close-btn"
        onClick={() => setIsMinimized(true)}
        title="Hide video"
      >
        ✕
      </button>

      {isPlaying ? (
        <div className="video-iframe-container">
          <iframe
            src={`https://www.youtube.com/embed/${videoId}?autoplay=1&rel=0`}
            title="Campus Tour Video"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
          />
        </div>
      ) : (
        <div
          className="video-poster"
          onClick={() => setIsPlaying(true)}
          role="button"
          tabIndex={0}
        >
          <div className="video-top-bar">
            <div className="video-channel-icon">
              <span className="channel-logo-text">🎓</span>
            </div>
            <div className="video-title-info">
              <h4>Experience the Difference: Where Practice Meets Theory</h4>
              <p>{collegeName}</p>
            </div>
            <div className="video-actions">
              <button type="button" className="icon-btn" title="Share">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M18 16.08c-.76 0-1.44.3-1.96.77L8.91 12.7c.05-.23.09-.46.09-.7s-.04-.47-.09-.7l7.05-4.11c.54.5 1.25.81 2.04.81 1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3c0 .24.04.47.09.7L8.04 9.81C7.5 9.31 6.79 9 6 9c-1.66 0-3 1.34-3 3s1.34 3 3 3c.79 0 1.5-.31 2.04-.81l7.12 4.16c-.05.21-.08.43-.08.65 0 1.61 1.31 2.92 2.92 2.92 1.61 0 2.92-1.31 2.92-2.92s-1.31-2.92-2.92-2.92z" />
                </svg>
              </button>
            </div>
          </div>

          <div className="video-center-content">
            <span className="college-highlight">CHOOSE {collegeName.toUpperCase()}</span>
            <div className="youtube-play-btn">
              <svg width="42" height="42" viewBox="0 0 68 48">
                <path
                  className="ytp-large-play-button-bg"
                  d="M66.52,7.74c-0.78-2.93-2.49-5.41-5.42-6.19C55.79,.13,34,0,34,0S12.21,.13,6.9,1.55 C3.97,2.33,2.27,4.81,1.48,7.74C0.06,13.05,0,24,0,24s0.06,10.95,1.48,16.26c0.78,2.93,2.49,5.41,5.42,6.19 C12.21,47.87,34,48,34,48s21.79-.13,27.1-1.55c2.93-.78,4.64-3.26,5.42-6.19C67.94,34.95,68,24,68,24S67.94,13.05,66.52,7.74z"
                  fill="#f00"
                />
                <path d="M 45,24 27,14 27,34" fill="#fff" />
              </svg>
            </div>
          </div>

          <div className="video-bottom-bar">
            <span className="watch-yt-badge">
              Watch on <strong>YouTube</strong>
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
