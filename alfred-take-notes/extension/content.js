chrome.runtime.onMessage.addListener(
(request,sender,response)=>{

if(request.type==="GET_VIDEO")
{

let url =
window.location.href;


let title =
document.title;


response({
 url,
 title
});

}

});