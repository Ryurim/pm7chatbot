const clock = document.getElementById("clock");

function updateClock() {
    const date = new Date();
    const hours = String(date.getHours()).padStart(2, "0");
    const minutes = String(date.getMinutes()).padStart(2, "0");
    clock.innerText = `${hours} : ${minutes}`;
}

// 매 초마다 시간을 업데이트
setInterval(updateClock, 1000);

// 처음 실행 시에도 바로 시간 표시
updateClock();
