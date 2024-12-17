/* Creation du canvas */
const canvas = document.querySelector(".monCanevas");
const width = (canvas.width = window.innerWidth);
const height = (canvas.height = window.innerWidth*0.5);
const ctx = canvas.getContext("2d");

/* Fonction pour convertir des degrèes en radians */
function degToRad(degrees){
    return (degrees*Math.PI)/180;
}

/* Creation d'un demi cercle comme base du compteur*/
ctx.fillStyle = "rgb(60.38959990413401,34.49826140859945,0)";
ctx.beginPath();
ctx.arc(window.innerWidth/2,280,200,degToRad(180),degToRad(0),false);
ctx.fill();

/* Creation d'une ligne pointillee au milieu du demi-cercle*/
ctx.fillStyle = "rgb(255,255,255)";
ctx.beginPath();
ctx.setLineDash([5,5]);
ctx.moveTo(window.innerWidth/2, 280);
ctx.lineTo(window.innerWidth/2, 80);
ctx.lineWidth = 2;
ctx.stroke();

/* Creation de l'aguille du compteur */
ctx.strokeStyle = '#4488EE';
ctx.beginPath();
ctx.setLineDash([]);
ctx.moveTo(window.innerWidth/2, 280);
ctx.lineTo(window.innerWidth/2 + 8, 81);
ctx.lineWidth = 2;
ctx.stroke();
ctx.fillStyle = '#4488EE';
ctx.beginPath();
ctx.moveTo(window.innerWidth/2 + 8, 81);
ctx.lineTo(window.innerWidth/2 + -1.115942587945538, 100.63488581160595);
ctx.lineTo(window.innerWidth/2 + 15.438016084366563, 101.32990572605362);
ctx.lineTo(window.innerWidth/2 + 8, 81);
ctx.fill();

/* Creation d'un texte de chaque côté de l'aiguille */
ctx.fillStyle = 'rgb(0,255,0)';
ctx.font = "48px georgia";
ctx.fillText("On time",width/2 + 120,320);
ctx.fillStyle = 'rgb(255,0,0)';
ctx.font = "48px georgia";
ctx.fillText("Delayed",width/2 - 300 ,320);

/* Creation d'un texte au bout de l'aiguille */
ctx.fillStyle = "rgb(60.38959990413401,34.49826140859945,0)";
ctx.font = "48px georgia"
var textString = "51.34 %",
    textWidth = ctx.measureText(textString).width;
ctx.fillText("51.34 %",(width/2) - (textWidth / 2),60)
