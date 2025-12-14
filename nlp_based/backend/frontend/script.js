const messagesDiv = document.getElementById("messages");
const input = document.getElementById("query-input");
const sendBtn = document.getElementById("send-btn");

function appendMessage(text, sender="bot") {
    const msgDiv = document.createElement("div");
    msgDiv.classList.add("message", sender);
    msgDiv.innerHTML = text;
    messagesDiv.appendChild(msgDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

async function sendQuery(query) {
    appendMessage(`<b>You:</b> ${query}`, "user");
    input.value = "";
    try {
        const response = await fetch("/get_answer", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query })
        });
        const data = await response.json();

        appendMessage(`<b>Bot:</b> ${data.answer}`, "bot");

        if (data.suggestions && data.suggestions.length > 0) {
            const suggestionList = document.createElement("ul");
            suggestionList.classList.add("suggestions");
            data.suggestions.forEach(s => {
                const li = document.createElement("li");
                li.textContent = s[0];
                li.onclick = () => sendQuery(s[0]);
                suggestionList.appendChild(li);
            });
            messagesDiv.appendChild(suggestionList);
        }
    } catch (err) {
        appendMessage("<b>Bot:</b> Error contacting server.", "bot");
        console.error(err);
    }
}

sendBtn.addEventListener("click", () => {
    const query = input.value.trim();
    if (query) sendQuery(query);
});

input.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
        sendBtn.click();
    }
});
