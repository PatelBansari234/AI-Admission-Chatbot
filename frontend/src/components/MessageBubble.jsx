export default function MessageBubble({ role, text, pending }) {
  const classes = ["msg", role, pending ? "pending" : ""].filter(Boolean).join(" ");

  function renderFormattedText(content) {
    if (!content) return "";
    const parts = [];
    const linkRegex = /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)|(https?:\/\/[^\s]+)/g;
    let lastIndex = 0;
    let match;

    while ((match = linkRegex.exec(content)) !== null) {
      if (match.index > lastIndex) {
        parts.push(content.substring(lastIndex, match.index));
      }
      if (match[1] && match[2]) {
        parts.push(
          <a
            key={match.index}
            href={match[2]}
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: "#2563eb", textDecoration: "underline", fontWeight: 500 }}
          >
            {match[1]}
          </a>
        );
      } else if (match[3]) {
        parts.push(
          <a
            key={match.index}
            href={match[3]}
            target="_blank"
            rel="noopener noreferrer"
            style={{ color: "#2563eb", textDecoration: "underline", fontWeight: 500 }}
          >
            {match[3]}
          </a>
        );
      }
      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < content.length) {
      parts.push(content.substring(lastIndex));
    }

    return parts.length > 0 ? parts : content;
  }

  return (
    <div className={classes} style={{ whiteSpace: "pre-wrap" }}>
      {renderFormattedText(text)}
    </div>
  );
}
